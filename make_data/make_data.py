import os
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import json
import re
import unidecode
from dotenv import load_dotenv
import os


load_dotenv()#LOAD MOI TRUONG

def chuan_hoa_1(s): #để làm tên file local
    s = re.sub(r'\W+', '', s)  # Xóa tất cả ký tự không phải chữ cái hoặc số
    s = unidecode.unidecode(s)  # Loại bỏ dấu tiếng Việt
    return s.lower()  # Chuyển thành chữ thường
def chuan_hoa_2(text):# để chuẩn hóa nội dung văn bản luật
        
    # Loại bỏ dấu xuống dòng không cần thiết giữa các từ
    text = re.sub(r"(\S)\r?\n\s*(\S)", r"\1 \2", text)

    # Xóa khoảng trắng dư thừa
    text = re.sub(r"\s{2,}", " ", text)

    # Loại bỏ dấu "-------" hoặc các dấu gạch không cần thiết
    text = re.sub(r"[-]{3,}", "", text)

    return text


class MakeData():
    def __init__(self):
        self.luat = {}


    def __get_content(self, link:str):
        try: 
            response = requests.get(link)
            soup = BeautifulSoup(response.content, 'html.parser').find('div', 'content1')
            p_tags = soup.find_all("p")
            
            # lấy thông tin luật
            luat_so = p_tags[2].get_text()
            co_quan_ban_hanh = p_tags[0].get_text()
            ngay_ban_hanh = p_tags[3].get_text()
            ten_luat = p_tags[6].get_text()
            self.file_name = chuan_hoa_1(ten_luat)

            # lấy nội dung luật
            chapters = {}  # Lưu chương và các điều luật bên trong
            current_chapter = None
            previous_chapter = None
            current_article = None
            flag = True
            for p in p_tags[9:]:
                try:
                    text = p.get_text(strip=True)  # Chuẩn hóa văn bản

                    # Nếu là tiêu đề chương
                    if text.startswith("Chương"):
                        previous_chapter = current_chapter
                        current_chapter = text
                        if current_chapter != previous_chapter:
                            flag = True


                    # Nếu là điều luật
                    elif text.startswith("Điều"):
                        current_article = chuan_hoa_2(text)
                        chapters[current_chapter][current_article] = []

                    # Nội dung điều luật
                    elif current_chapter:
                        if flag:
                            current_chapter += ' ' + text
                            current_chapter = chuan_hoa_2(current_chapter)
                            chapters[current_chapter] = {}
                            flag = False
                        else:
                            if text[1] == ')': #tưc là chỉ mục ví dụ như a), b)
                                chapters[current_chapter][current_article][-1] += chuan_hoa_2(f'mục {text}\n')
                            else: 
                                chapters[current_chapter][current_article].append( chuan_hoa_2(f'khoản {text}\n'))
                            
                    

                except Exception as e:
                    print(f"❌ Lỗi khi xử lý thẻ <p>: {e}")
                    continue  # Bỏ qua lỗi, tiếp tục vòng lặp
                
            self.luat['ten_luat'] = chuan_hoa_2(ten_luat)
            self.luat['co_quan_ban_hanh'] = chuan_hoa_2(co_quan_ban_hanh)
            self.luat['luat_so'] = chuan_hoa_2(luat_so)
            self.luat['ngay_ban_hanh'] = chuan_hoa_2(ngay_ban_hanh)
            self.luat['ten_luat'] = chuan_hoa_2('LUẬT ' + ten_luat)
            self.luat['noi_dung'] = chapters

        except:
            print('ko truy cập đc')

    def store_at_local(self, link:str):
        self.__get_content(link)
        directory_path = os.getenv('DIRECTORY_PATH')
        self.file_path = f'{directory_path}{self.file_name}.json'
        
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.luat, f, ensure_ascii=False, indent=4)
        print('đã lưu xog')
    
    def insert_into_db(self, link):
        self.__get_content(link)
        try:
            uri = os.getenv('URI')
            client = MongoClient(uri)
            db = client[os.getenv('DB_NAME')]
            collection = db[os.getenv('COLLECTION_NAME')]
        except:
            print('lỗi do chưa chạy db')
        collection.insert_one(self.luat)
        print('xong')
        
    
        

    
if __name__ == '__main__':
    makedata = MakeData()
    makedata.store_at_local('https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-trat-tu-an-toan-giao-thong-duong-bo-2024-so-36-2024-QH15-444251.aspx')