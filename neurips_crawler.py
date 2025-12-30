import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 설정 ---
START_YEAR = 1987
END_YEAR = 2024
BASE_URL = "https://papers.nips.cc"
OUTPUT_DIR = "neurips_data_full_desc"  # 저장 폴더명 변경

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 봇 차단 방지 헤더
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
}

def clean_text(text):
    """줄바꿈과 불필요한 공백을 제거하여 한 줄로 만듭니다."""
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_abstract_from_semantic_scholar(title):
    """Semantic Scholar API를 통해 Abstract 검색"""
    if not title:
        return ""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {'query': title, 'limit': 1, 'fields': 'abstract'}
        response = requests.get(url, params=params, headers=HEADERS, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data'] and data['data'][0].get('abstract'):
                return clean_text(data['data'][0]['abstract'])
    except Exception:
        pass
    return ""

def get_paper_details(paper_info):
    """개별 논문 상세 정보 수집"""
    paper_url = paper_info['url']
    year = paper_info['year']
    
    try:
        response = requests.get(paper_url, headers=HEADERS, timeout=20)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Title
        title_tag = soup.find('title')
        title = clean_text(title_tag.text) if title_tag else ""
        
        # 2. Abstract
        abstract_text = ""
        
        # [우선순위 1] Meta 태그 확인 (가장 정확)
        meta_abstract = soup.find('meta', attrs={'name': 'citation_abstract'})
        if meta_abstract and meta_abstract.get('content'):
            abstract_text = clean_text(meta_abstract['content'])
        
        # [우선순위 2] HTML 본문 헤더 검색
        if not abstract_text:
            headers = soup.find_all(['h4', 'h3', 'h5', 'h2'], string=re.compile(r'Abstract', re.IGNORECASE))
            for h in headers:
                next_elem = h.find_next_sibling()
                if next_elem and next_elem.name == 'p':
                    abstract_text = clean_text(next_elem.text)
                    if abstract_text: break
        
        # [우선순위 3] Semantic Scholar 검색
        if not abstract_text and title:
            abstract_text = get_abstract_from_semantic_scholar(title)
            if not abstract_text:
                abstract_text = "Abstract Not Available"

        # 3. BibTeX
        bibtex_link_tag = soup.find('a', string=re.compile(r'BibTeX', re.IGNORECASE))
        bibtex_text = ""
        
        if bibtex_link_tag:
            bibtex_url = BASE_URL + bibtex_link_tag['href']
            try:
                bib_response = requests.get(bibtex_url, headers=HEADERS, timeout=10)
                if bib_response.status_code == 200:
                    bibtex_text = clean_text(bib_response.text)
            except:
                bibtex_text = "Failed to fetch BibTeX"

        return {
            'year': year,
            'title': title,
            'abstract': abstract_text,
            'bibtex': bibtex_text,
            'url': paper_url
        }

    except Exception:
        return None

def process_year(year):
    """해당 연도의 전체 논문 리스트 수집"""
    list_url = f"{BASE_URL}/paper_files/paper/{year}"
    
    # 잦은 요청 시 연결 끊김 방지를 위한 재시도 로직
    for attempt in range(3):
        try:
            response = requests.get(list_url, headers=HEADERS, timeout=30)
            if response.status_code == 200:
                break
        except:
            time.sleep(2)
    else:
        print(f"[{year}] 페이지 접근 실패. 건너뜁니다.")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    paper_links = soup.select('ul.paper-list li a[title="paper title"]')
    if not paper_links:
        paper_links = soup.select('ul.paper-list li a')

    valid_links = []
    for link in paper_links:
        if 'href' in link.attrs and '/paper_files/paper/' in link['href']:
            if not link['href'].endswith('.pdf'):
                valid_links.append(BASE_URL + link['href'])
    
    valid_links = list(set(valid_links))
    
    # ---------------------------------------------------------
    # [수정] 제한 슬라이싱([:50])을 제거하여 전체 수집
    # ---------------------------------------------------------
    print(f"[{year}] 총 {len(valid_links)}개의 논문 수집 시작...")
    
    results = []
    
    # 병렬 처리 (max_workers=5 권장)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_paper_details, {'url': url, 'year': year}) for url in valid_links]
        
        for future in tqdm(as_completed(futures), total=len(valid_links), desc=f"Year {year}"):
            data = future.result()
            if data:
                results.append(data)
    
    if results:
        df = pd.DataFrame(results)
        df = df[['year', 'title', 'abstract', 'bibtex', 'url']]
        filename = f"{OUTPUT_DIR}/neurips_{year}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"[{year}] 저장 완료: {filename} ({len(results)}건)")
    else:
        print(f"[{year}] 수집된 데이터가 없습니다.")

def main():
    print(f"NeurIPS {END_YEAR} ~ {START_YEAR} 전체 데이터 수집 시작 (역순)")
    
    # [수정] 2024부터 1987까지 역순으로 반복
    for year in range(END_YEAR, START_YEAR - 1, -1):
        process_year(year)
        print(f"[{year}] 완료. 3초 대기 후 다음 연도 진행...")
        time.sleep(3) # 연도 간 대기 시간

if __name__ == "__main__":
    main()