"""
eCourts Judgments Scraper
Scrapes judgments from https://judgments.ecourts.gov.in

This script uses Selenium for browser automation since the website:
1. Requires captcha verification
2. Has dynamic content loaded via JavaScript
3. Has anti-bot protection

Usage:
    python scraper.py --search "contract dispute" --court "Supreme Court"
    python scraper.py --search "property law" --max-results 50
"""

import os
import time
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait, Select
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not installed. Run: pip install selenium")

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Requests/BeautifulSoup not installed. Run: pip install requests beautifulsoup4")


# Configuration
BASE_URL = "https://judgments.ecourts.gov.in/pdfsearch/"
OUTPUT_DIR = "./scraped_judgments"
RESULTS_FILE = "judgment_results.json"


def setup_driver(headless=False):
    """Setup Chrome WebDriver with appropriate options"""
    options = Options()
    
    if headless:
        options.add_argument("--headless")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Disable automation flags
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=options)
    
    # Remove webdriver property
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver


def wait_for_captcha_solution(driver, timeout=120):
    """
    Wait for user to manually solve captcha
    Returns True if captcha was solved, False if timeout
    """
    print("\n" + "="*50)
    print("CAPTCHA DETECTED!")
    print("Please solve the captcha in the browser window.")
    print(f"Waiting up to {timeout} seconds...")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check if we're past the captcha (search results or search form visible)
            search_box = driver.find_elements(By.ID, "search_text")
            results = driver.find_elements(By.CLASS_NAME, "search-result")
            
            if search_box or results:
                print("Captcha solved! Continuing...")
                return True
        except:
            pass
        
        time.sleep(1)
    
    print("Captcha timeout. Please try again.")
    return False


def search_judgments(driver, search_query, court_type="2", max_results=100):
    """
    Search for judgments on eCourts
    
    Args:
        driver: Selenium WebDriver
        search_query: Text to search for
        court_type: "1" for District Courts, "2" for High Courts
        max_results: Maximum number of results to scrape
    
    Returns:
        List of judgment dictionaries
    """
    judgments = []
    
    try:
        # Navigate to the search page
        driver.get(BASE_URL)
        time.sleep(3)
        
        # Wait for and handle captcha if present
        captcha_elements = driver.find_elements(By.ID, "captcha")
        if captcha_elements:
            if not wait_for_captcha_solution(driver):
                return judgments
        
        # Wait for the search form
        wait = WebDriverWait(driver, 20)
        
        try:
            search_box = wait.until(EC.presence_of_element_located((By.ID, "search_text")))
        except TimeoutException:
            # Try alternative selectors
            search_box = wait.until(EC.presence_of_element_located((By.NAME, "text")))
        
        # Clear and enter search query
        search_box.clear()
        search_box.send_keys(search_query)
        
        # Select court type if dropdown exists
        try:
            court_select = Select(driver.find_element(By.ID, "fcourt_type"))
            court_select.select_by_value(court_type)
        except NoSuchElementException:
            print("Court type selector not found, using default")
        
        # Handle captcha input if required
        try:
            captcha_input = driver.find_element(By.ID, "captcha")
            if captcha_input.is_displayed():
                print("\nPlease enter the captcha manually in the browser...")
                time.sleep(30)  # Wait for manual captcha entry
        except NoSuchElementException:
            pass
        
        # Click search button
        try:
            search_btn = driver.find_element(By.ID, "search_btn")
            search_btn.click()
        except NoSuchElementException:
            # Try submitting the form
            search_box.send_keys(Keys.RETURN)
        
        time.sleep(5)
        
        # Scrape results
        page = 1
        while len(judgments) < max_results:
            print(f"Scraping page {page}...")
            
            # Wait for results to load
            time.sleep(3)
            
            # Find all judgment entries
            result_items = driver.find_elements(By.CSS_SELECTOR, ".search-result, .result-item, tr.result-row")
            
            if not result_items:
                # Try alternative selectors
                result_items = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            
            if not result_items:
                print("No results found on this page")
                break
            
            for item in result_items:
                if len(judgments) >= max_results:
                    break
                
                try:
                    judgment = extract_judgment_info(item, driver)
                    if judgment:
                        judgments.append(judgment)
                        print(f"  Found: {judgment.get('title', 'Unknown')[:50]}...")
                except Exception as e:
                    print(f"  Error extracting judgment: {e}")
                    continue
            
            # Try to go to next page
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, ".next-page, .pagination a.next, a[rel='next']")
                if next_btn.is_enabled():
                    next_btn.click()
                    page += 1
                    time.sleep(3)
                else:
                    break
            except NoSuchElementException:
                print("No more pages")
                break
    
    except Exception as e:
        print(f"Error during search: {e}")
    
    return judgments


def extract_judgment_info(element, driver):
    """Extract judgment information from a search result element"""
    judgment = {
        "scraped_at": datetime.now().isoformat(),
    }
    
    # Try to extract various fields
    # These selectors may need adjustment based on actual page structure
    
    # Title/Case Name
    try:
        title_elem = element.find_element(By.CSS_SELECTOR, ".case-title, .title, a, td:first-child")
        judgment["title"] = title_elem.text.strip()
    except:
        judgment["title"] = element.text[:100] if element.text else "Unknown"
    
    # Case Number
    try:
        case_num = element.find_element(By.CSS_SELECTOR, ".case-number, .case-no")
        judgment["case_number"] = case_num.text.strip()
    except:
        pass
    
    # Court Name
    try:
        court = element.find_element(By.CSS_SELECTOR, ".court-name, .court")
        judgment["court"] = court.text.strip()
    except:
        pass
    
    # Date
    try:
        date_elem = element.find_element(By.CSS_SELECTOR, ".date, .judgment-date")
        judgment["date"] = date_elem.text.strip()
    except:
        pass
    
    # PDF Link
    try:
        pdf_link = element.find_element(By.CSS_SELECTOR, "a[href*='.pdf'], a[href*='download']")
        judgment["pdf_url"] = pdf_link.get_attribute("href")
    except:
        pass
    
    # Detail page link
    try:
        detail_link = element.find_element(By.CSS_SELECTOR, "a")
        judgment["detail_url"] = detail_link.get_attribute("href")
    except:
        pass
    
    return judgment if judgment.get("title") else None


def download_pdf(url, filename, output_dir=OUTPUT_DIR):
    """Download a PDF file"""
    if not REQUESTS_AVAILABLE:
        print("Requests not available, cannot download PDFs")
        return None
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return filepath
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def save_results(judgments, filename=RESULTS_FILE, format="json"):
    """Save scraped results to file"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if format == "json":
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(judgments, f, indent=2, ensure_ascii=False)
    elif format == "csv":
        if judgments:
            keys = set()
            for j in judgments:
                keys.update(j.keys())
            
            with open(filepath.replace('.json', '.csv'), 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(keys))
                writer.writeheader()
                writer.writerows(judgments)
    
    print(f"Results saved to: {filepath}")
    return filepath


def interactive_mode():
    """Run scraper in interactive mode with user prompts"""
    print("\n" + "="*60)
    print("eCourts Judgment Scraper - Interactive Mode")
    print("="*60)
    
    if not SELENIUM_AVAILABLE:
        print("\nError: Selenium is required. Install it with:")
        print("  pip install selenium")
        print("\nYou also need Chrome and ChromeDriver installed.")
        return
    
    search_query = input("\nEnter search query (e.g., 'contract dispute'): ").strip()
    if not search_query:
        print("No query provided. Exiting.")
        return
    
    court_type = input("Select court type [1=District, 2=High Court] (default: 2): ").strip() or "2"
    max_results = int(input("Maximum results to scrape (default: 50): ").strip() or "50")
    headless = input("Run in headless mode? [y/N]: ").strip().lower() == 'y'
    
    print("\nStarting browser...")
    driver = setup_driver(headless=headless)
    
    try:
        print(f"Searching for: '{search_query}'")
        judgments = search_judgments(driver, search_query, court_type, max_results)
        
        print(f"\nFound {len(judgments)} judgments")
        
        if judgments:
            save_results(judgments)
            save_results(judgments, format="csv")
            
            download_pdfs = input("\nDownload PDFs? [y/N]: ").strip().lower() == 'y'
            if download_pdfs:
                for i, j in enumerate(judgments):
                    if j.get("pdf_url"):
                        filename = f"judgment_{i+1}.pdf"
                        download_pdf(j["pdf_url"], filename)
    
    finally:
        driver.quit()
        print("\nScraping complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Scrape judgments from eCourts India")
    parser.add_argument("--search", "-s", type=str, help="Search query")
    parser.add_argument("--court", "-c", type=str, default="2", 
                       choices=["1", "2"], help="Court type: 1=District, 2=High Court")
    parser.add_argument("--max-results", "-m", type=int, default=50, 
                       help="Maximum results to scrape")
    parser.add_argument("--headless", action="store_true", 
                       help="Run browser in headless mode")
    parser.add_argument("--download-pdfs", action="store_true", 
                       help="Download PDF files")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.search:
        interactive_mode()
        return
    
    if not SELENIUM_AVAILABLE:
        print("Error: Selenium is required. Install with: pip install selenium")
        return
    
    print(f"Starting scraper for: '{args.search}'")
    driver = setup_driver(headless=args.headless)
    
    try:
        judgments = search_judgments(driver, args.search, args.court, args.max_results)
        
        print(f"\nFound {len(judgments)} judgments")
        
        if judgments:
            save_results(judgments)
            save_results(judgments, format="csv")
            
            if args.download_pdfs:
                for i, j in enumerate(judgments):
                    if j.get("pdf_url"):
                        filename = f"judgment_{i+1}.pdf"
                        download_pdf(j["pdf_url"], filename)
    
    finally:
        driver.quit()
        print("\nDone!")


if __name__ == "__main__":
    main()
