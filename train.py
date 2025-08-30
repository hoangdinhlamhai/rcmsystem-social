import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer

# charset=utf8 để xử lý tiếng Việt
DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/social?charset=utf8"
engine = create_engine(DATABASE_URL)

# GROUP_CONCAT dùng để nối các chuỗi
query = """
SELECT 
    u.id, 
    u.first_name, 
    u.last_name,
    GROUP_CONCAT(s.skill_name) AS skills
FROM 
    users u
LEFT JOIN 
    skills s ON u.id = s.user_id
WHERE 
    s.skill_name IS NOT NULL
GROUP BY 
    u.id;
"""

print("Đang tải dữ liệu từ database MySQL...")
df = pd.read_sql(query, engine)
df = df.fillna('') # Xử lý các giá trị NULL
print(f"Đã tải {len(df)} người dùng có kỹ năng.")
print("Dữ liệu ví dụ:")
print(df.head())

# --- 2. FEATURE ENGINEERING: TẠO EMBEDDINGS ---
print("Đang tải mô hình AI...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

df.rename(columns={'skills': 'profile_text'}, inplace=True)
print("\nVí dụ về profile_text được tạo ra (chỉ từ skills):")
print(df[['id', 'profile_text']].head())

print("Đang tạo embeddings cho người dùng...")
# Dùng mô hình để biến đổi cột profile_text (chứa skills) thành các vector số
user_embeddings = model.encode(df['profile_text'].tolist(), show_progress_bar=True)

# --- 3. LƯU CÁC "HIỆN VẬT" (ARTIFACTS) ---
print("Đang lưu các artifacts...")

# Tạo thư mục nếu nó chưa tồn tại (để tránh lỗi)
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')

np.save('artifacts/user_embeddings.npy', user_embeddings)

# Lưu ý: user_id_to_index bây giờ sẽ chỉ chứa ID của những user có skill
user_id_to_index = {str(user_id): index for index, user_id in enumerate(df['id'])}
with open('artifacts/user_id_to_index.json', 'w') as f:
    json.dump(user_id_to_index, f)
    
print("\nHoàn tất! Mô hình đã sẵn sàng để được triển khai.")

# file .npy lưu các vector chứa các đặc tính của user
# file .json như mục lục: user có id là x vector của nó nằm ở dòng y trong file .npy