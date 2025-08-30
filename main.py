import numpy as np
import json
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. KHỞI TẠO ỨNG DỤNG VÀ TẢI MÔ HÌNH ---
print("Đang khởi tạo AI service...")
app = FastAPI()

# Tải các artifacts đã được huấn luyện vào bộ nhớ khi server khởi động
user_embeddings = np.load('artifacts/user_embeddings.npy')
with open('artifacts/user_id_to_index.json', 'r') as f:
    user_id_to_index = json.load(f)

# Tạo một map ngược lại để dễ dàng tra cứu id từ index
index_to_user_id = {index: user_id for user_id, index in user_id_to_index.items()}

print("AI service đã sẵn sàng!")

# --- 2. ĐỊNH NGHĨA ENDPOINT GỢI Ý ---
@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, top_n: int = 5):
    """
    Nhận vào user_id và trả về top_n người dùng tương đồng nhất.
    """
    # Kiểm tra xem user_id có tồn tại trong dữ liệu đã train không
    if user_id not in user_id_to_index:
        raise HTTPException(status_code=404, detail="User ID not found in trained data")

    # Lấy index của người dùng hiện tại
    current_user_index = user_id_to_index[user_id]
    
    # Lấy vector embedding của người dùng hiện tại
    current_user_embedding = user_embeddings[current_user_index].reshape(1, -1)
    
    # --- PHẦN TÍNH TOÁN CỐT LÕI ---
    # Tính độ tương đồng cosine giữa vector của người dùng hiện tại và TẤT CẢ các vector khác
    similarities = cosine_similarity(current_user_embedding, user_embeddings)[0]
    
    # Lấy ra index của N người dùng có độ tương đồng cao nhất (trừ chính người đó)
    # np.argsort trả về index của các phần tử sau khi đã sắp xếp
    most_similar_indices = np.argsort(similarities)[-top_n-1:-1][::-1]
    
    # Chuyển đổi từ index trở lại user_id
    recommended_user_ids = [index_to_user_id[idx] for idx in most_similar_indices]
    
    return {"recommended_user_ids": recommended_user_ids}