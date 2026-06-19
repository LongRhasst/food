"""
Hệ thống Gợi ý Món ăn Cá nhân hóa bằng RFM + K-Means
=======================================================
RFM-based Collaborative Filtering:
  "Phân nhóm khách hàng theo Recency–Frequency–Monetary,
   kết hợp lịch sử đặt hàng cá nhân để gợi ý món ăn phù hợp."

Pipeline:
  1. Đọc dữ liệu hóa đơn từ food.csv (sinh bởi gen_data.py)
  2. Tính chỉ số RFM (Recency, Frequency, Monetary) cho mỗi khách
  3. Chuẩn hóa RFM → K-Means phân cụm
  4. Gợi ý món ăn theo 3 tầng:
     Tầng 1: Lịch sử cá nhân (món đặt nhiều, phù hợp túi tiền)
     Tầng 2: Xu hướng cụm (món phổ biến trong cụm, chưa thử)
     Tầng 3: Best-sellers toàn hệ thống (cold-start)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================
#  PHẦN 1: ĐỌC DỮ LIỆU TỪ food.csv
# ============================================================

def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Đọc dữ liệu hóa đơn từ file food.csv.

    File food.csv được sinh bởi gen_data.py với cấu trúc:
        InvoiceNo, InvoiceDate, CustomerID, FoodID, FoodName, Quantity, UnitPrice
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "food.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Khong tim thay file: {csv_path}\n"
            f"Hay chay 'python gen_data.py' truoc de sinh du lieu."
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["InvoiceDate"])

    # Tính tổng tiền mỗi dòng
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    print(f"Da doc {len(df):,} dong du lieu tu {os.path.basename(csv_path)}")
    print(f"   So khach hang  : {df['CustomerID'].nunique():,}")
    print(f"   So mon an      : {df['FoodID'].nunique()}")
    print(f"   So hoa don     : {df['InvoiceNo'].nunique():,}")
    print(f"   Khoang thoi gian: {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}")
    print(f"   Tong doanh thu : {df['TotalAmount'].sum():,.0f} VND")
    return df


# ============================================================
#  PHẦN 2: TÍNH CHỈ SỐ RFM
# ============================================================

def compute_rfm(df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    Tính Recency–Frequency–Monetary cho mỗi khách hàng.

    - Recency:   Số ngày từ lần mua gần nhất → ngày tham chiếu
    - Frequency: Số hóa đơn (đơn hàng) riêng biệt
    - Monetary:  Tổng chi tiêu (Quantity × UnitPrice)

    Returns:
        DataFrame với index = CustomerID, columns = [Recency, Frequency, Monetary]
    """
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    )

    print(f"\n📊 RFM Summary:")
    print(f"   Recency  — min: {rfm['Recency'].min()}, median: {rfm['Recency'].median():.0f}, max: {rfm['Recency'].max()}")
    print(f"   Frequency— min: {rfm['Frequency'].min()}, median: {rfm['Frequency'].median():.0f}, max: {rfm['Frequency'].max()}")
    print(f"   Monetary — min: {rfm['Monetary'].min():,.0f}, median: {rfm['Monetary'].median():,.0f}, max: {rfm['Monetary'].max():,.0f}")

    return rfm


# ============================================================
#  PHẦN 3: XÂY DỰNG PIPELINE GỢI Ý
# ============================================================

class FoodRecommender:
    """
    Hệ thống gợi ý món ăn dựa trên RFM + K-Means Clustering.

    Pipeline:
        1. Tính RFM cho mỗi khách hàng
        2. Chuẩn hóa RFM (StandardScaler)
        3. K-Means phân cụm trên vector RFM
        4. Xây dựng order history & cluster food popularity
        5. Gợi ý 3 tầng: Cá nhân → Cụm → Best-sellers
    """

    def __init__(
        self,
        n_clusters: int = 5,
        cold_start_threshold: int = 2,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.cold_start_threshold = cold_start_threshold
        self.random_state = random_state

        # Kết quả sau khi fit
        self.kmeans_model = None
        self.scaler = None
        self.rfm = None                          # DataFrame RFM
        self.rfm_scaled = None                   # Mảng RFM đã chuẩn hóa
        self.cluster_labels = None
        self.customer_cluster_map = None         # {CustomerID: cluster}
        self.food_id_to_name = None              # {FoodID: FoodName}
        self.food_id_to_price = None             # {FoodID: avg UnitPrice}
        self.order_history = None                # {CustomerID: {FoodID: {count, total_qty, total_spend}}}
        self.customer_avg_price = None           # {CustomerID: avg price per item}
        self.cluster_food_popularity = None      # {cluster: DataFrame(FoodID, order_count, total_qty)}
        self.bestsellers = None                  # Top best-selling FoodIDs
        self.df = None                           # Raw data reference

    def fit(self, df: pd.DataFrame):
        """Huấn luyện mô hình trên toàn bộ dữ liệu."""

        print("\n" + "=" * 60)
        print("  🔧 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH GỢI Ý (RFM + K-Means)")
        print("=" * 60)

        self.df = df

        # --- Lưu mapping foodID → foodName, foodID → price ---
        self.food_id_to_name = (
            df.drop_duplicates("FoodID")
            .set_index("FoodID")["FoodName"]
            .to_dict()
        )
        self.food_id_to_price = (
            df.groupby("FoodID")["UnitPrice"]
            .mean()
            .to_dict()
        )

        # --- Bước 1: Tính RFM ---
        print(f"\n📅 Bước 1 – Tính chỉ số RFM")
        self.rfm = compute_rfm(df)

        # --- Bước 2: Chuẩn hóa RFM ---
        self.scaler = StandardScaler()
        self.rfm_scaled = self.scaler.fit_transform(self.rfm.values)

        print(f"\n🔢 Bước 2 – Chuẩn hóa RFM (StandardScaler)")
        print(f"   Kích thước: {self.rfm_scaled.shape[0]:,} khách × {self.rfm_scaled.shape[1]} features (R, F, M)")

        # --- Bước 3: K-Means trên RFM ---
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )
        self.cluster_labels = self.kmeans_model.fit_predict(self.rfm_scaled)

        self.customer_cluster_map = dict(
            zip(self.rfm.index, self.cluster_labels)
        )

        # Gán cụm vào RFM DataFrame
        self.rfm["Cluster"] = self.cluster_labels

        print(f"\n🎯 Bước 3 – K-Means Clustering trên RFM")
        print(f"   Số cụm (K): {self.n_clusters}")
        sil_score = silhouette_score(self.rfm_scaled, self.cluster_labels)
        print(f"   Silhouette Score: {sil_score:.3f}")
        for c in range(self.n_clusters):
            count = (self.cluster_labels == c).sum()
            cluster_rfm = self.rfm[self.rfm["Cluster"] == c]
            print(f"   • Cụm {c}: {count:,} khách | "
                  f"R={cluster_rfm['Recency'].median():.0f}d | "
                  f"F={cluster_rfm['Frequency'].median():.0f} đơn | "
                  f"M={cluster_rfm['Monetary'].median():,.0f}đ")

        # --- Bước 4: Xây dựng Order History ---
        print(f"\n📦 Bước 4 – Xây dựng lịch sử đặt hàng & thống kê cụm")
        self._build_order_history(df)
        self._build_cluster_food_popularity(df)
        self._build_bestsellers(df)

        print(f"\n✅ HUẤN LUYỆN HOÀN TẤT!")
        return self

    def _build_order_history(self, df: pd.DataFrame):
        """Xây dựng lịch sử đặt hàng chi tiết cho mỗi khách."""
        # {CustomerID: {FoodID: {count: int, total_qty: int, total_spend: float}}}
        self.order_history = defaultdict(lambda: defaultdict(lambda: {
            "count": 0, "total_qty": 0, "total_spend": 0
        }))

        # Tính avg price per item cho mỗi khách
        customer_stats = df.groupby("CustomerID").agg(
            total_spend=("TotalAmount", "sum"),
            total_items=("Quantity", "sum"),
        )
        self.customer_avg_price = (customer_stats["total_spend"] / customer_stats["total_items"]).to_dict()

        # Aggregate order history
        order_agg = df.groupby(["CustomerID", "FoodID"]).agg(
            count=("InvoiceNo", "nunique"),
            total_qty=("Quantity", "sum"),
            total_spend=("TotalAmount", "sum"),
        ).reset_index()

        for _, row in order_agg.iterrows():
            self.order_history[row["CustomerID"]][row["FoodID"]] = {
                "count": int(row["count"]),
                "total_qty": int(row["total_qty"]),
                "total_spend": float(row["total_spend"]),
            }

    def _build_cluster_food_popularity(self, df: pd.DataFrame):
        """Tính món phổ biến trong từng cụm RFM."""
        # Gắn cluster vào dữ liệu gốc
        df_with_cluster = df.merge(
            self.rfm[["Cluster"]],
            left_on="CustomerID",
            right_index=True,
            how="inner",
        )

        self.cluster_food_popularity = {}
        for c in range(self.n_clusters):
            cluster_data = df_with_cluster[df_with_cluster["Cluster"] == c]
            food_pop = cluster_data.groupby("FoodID").agg(
                order_count=("InvoiceNo", "nunique"),
                total_qty=("Quantity", "sum"),
                customer_count=("CustomerID", "nunique"),
                avg_price=("UnitPrice", "mean"),
            ).sort_values("customer_count", ascending=False)
            self.cluster_food_popularity[c] = food_pop

    def _build_bestsellers(self, df: pd.DataFrame):
        """Tạo danh sách best-sellers toàn hệ thống."""
        self.bestsellers = (
            df.groupby("FoodID")
            .agg(
                customer_count=("CustomerID", "nunique"),
                order_count=("InvoiceNo", "nunique"),
            )
            .sort_values("customer_count", ascending=False)
            .head(15)
            .index.tolist()
        )

    def recommend(self, customer_id: str, top_k: int = 5) -> pd.DataFrame:
        """
        Gợi ý top_k món ăn cho 1 khách hàng.

        Logic 3 tầng:
          Tầng 1: Cá nhân — món đặt nhiều trong lịch sử, phù hợp túi tiền
          Tầng 2: Cụm    — món phổ biến trong cụm RFM mà khách chưa thử
          Tầng 3: Fallback— Best-sellers (cold-start hoặc hết gợi ý)
        """
        result_rows = []

        # --- Cold-Start: Khách mới chưa có dữ liệu ---
        if customer_id not in self.customer_cluster_map:
            print(f"❄️  Khách [{customer_id}] → COLD-START → Gợi ý Best-Sellers")
            for rank, fid in enumerate(self.bestsellers[:top_k], 1):
                result_rows.append({
                    "rank": rank,
                    "foodID": fid,
                    "foodName": self.food_id_to_name.get(fid, ""),
                    "unitPrice": f"{self.food_id_to_price.get(fid, 0):,.0f}đ",
                    "reason": "🏆 Best-Seller (Khách mới)",
                    "score_detail": "N/A",
                })
            return pd.DataFrame(result_rows)

        # --- Lấy thông tin khách ---
        cluster_id = self.customer_cluster_map[customer_id]
        history = self.order_history.get(customer_id, {})
        avg_price = self.customer_avg_price.get(customer_id, 100_000)
        price_ceiling = avg_price * 1.5  # Ngưỡng giá phù hợp

        rfm_info = self.rfm.loc[customer_id]
        print(f"Khách [{customer_id}] → Cụm {cluster_id} | "
              f"R={rfm_info['Recency']:.0f}d F={rfm_info['Frequency']:.0f} M={rfm_info['Monetary']:,.0f}đ | "
              f"Mức giá TB: {avg_price:,.0f}đ")

        eaten_food_ids = set(history.keys())

        # =============================================
        #  TẦNG 1: Gợi ý từ lịch sử cá nhân
        #           "Những món bạn hay đặt"
        # =============================================
        if len(history) >= self.cold_start_threshold:
            # Sắp xếp theo số lần đặt (count) giảm dần
            personal_favorites = sorted(
                history.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )

            for fid, stats in personal_favorites:
                if len(result_rows) >= top_k:
                    break
                price = self.food_id_to_price.get(fid, 0)
                result_rows.append({
                    "rank": len(result_rows) + 1,
                    "foodID": fid,
                    "foodName": self.food_id_to_name.get(fid, ""),
                    "unitPrice": f"{price:,.0f}đ",
                    "reason": f"⭐ Yêu thích cá nhân (đặt {stats['count']} lần)",
                    "score_detail": f"Qty={stats['total_qty']}, Spend={stats['total_spend']:,.0f}đ",
                })

        # =============================================
        #  TẦNG 2: Gợi ý từ cụm RFM
        #           "Khách giống bạn cũng thích"
        # =============================================
        if len(result_rows) < top_k:
            cluster_pop = self.cluster_food_popularity.get(cluster_id)
            if cluster_pop is not None:
                already_suggested = {r["foodID"] for r in result_rows}

                for fid, row in cluster_pop.iterrows():
                    if len(result_rows) >= top_k:
                        break
                    if fid in already_suggested:
                        continue
                    # Lọc theo khả năng chi trả
                    item_price = self.food_id_to_price.get(fid, 0)
                    if item_price > price_ceiling:
                        continue

                    # Ưu tiên món khách chưa thử
                    is_new = fid not in eaten_food_ids
                    label = "🔍 Khám phá mới từ cụm" if is_new else "📈 Phổ biến trong cụm"

                    result_rows.append({
                        "rank": len(result_rows) + 1,
                        "foodID": fid,
                        "foodName": self.food_id_to_name.get(fid, ""),
                        "unitPrice": f"{item_price:,.0f}đ",
                        "reason": f"{label} {cluster_id}",
                        "score_detail": f"{int(row['customer_count'])} khách đặt, phù hợp mức giá",
                    })

        # =============================================
        #  TẦNG 3: Fallback — Best-sellers
        # =============================================
        if len(result_rows) < top_k:
            already_suggested = {r["foodID"] for r in result_rows}
            for fid in self.bestsellers:
                if len(result_rows) >= top_k:
                    break
                if fid in already_suggested:
                    continue
                price = self.food_id_to_price.get(fid, 0)
                result_rows.append({
                    "rank": len(result_rows) + 1,
                    "foodID": fid,
                    "foodName": self.food_id_to_name.get(fid, ""),
                    "unitPrice": f"{price:,.0f}đ",
                    "reason": "🏆 Best-Seller hệ thống",
                    "score_detail": "Top toàn hệ thống",
                })

        return pd.DataFrame(result_rows)

    def recommend_batch(self, customer_ids: list, top_k: int = 5) -> dict:
        """Gợi ý cho nhiều khách hàng cùng lúc (Batch Processing)."""
        results = {}
        for cid in customer_ids:
            results[cid] = self.recommend(cid, top_k)
        return results


# ============================================================
#  PHẦN 4: TRỰC QUAN HÓA & ĐÁNH GIÁ
# ============================================================

def find_optimal_k(rfm_scaled: np.ndarray, k_range=range(2, 11), random_state=42):
    """
    Tìm số cụm K tối ưu bằng Elbow Method + Silhouette Score.
    """
    inertias = []
    silhouettes = []

    print("\n🔍 Tìm K tối ưu bằng Elbow + Silhouette...")
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    max_iter=300, random_state=random_state)
        labels = km.fit_predict(rfm_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(rfm_scaled, labels)
        silhouettes.append(sil)
        print(f"   K={k}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.3f}")

    # Tìm K tối ưu bằng Elbow (điểm xa đường thẳng nối hai đầu nhất)
    k_list = list(k_range)
    x1, y1 = k_list[0], inertias[0]
    x2, y2 = k_list[-1], inertias[-1]

    distances = []
    for i, k in enumerate(k_list):
        dist = abs((y2 - y1) * k - (x2 - x1) * inertias[i] + x2 * y1 - y2 * x1)
        distances.append(dist)

    elbow_k = k_list[np.argmax(distances)]
    best_sil_k = k_list[np.argmax(silhouettes)]

    # Vẽ biểu đồ Elbow + Silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow
    ax1.plot(k_list, inertias, "bo-", linewidth=2, markersize=8)
    ax1.axvline(x=elbow_k, color="green", linestyle="--", label=f"Elbow K = {elbow_k}")
    ax1.set_xlabel("Số cụm K", fontsize=12)
    ax1.set_ylabel("Inertia (SSE)", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Silhouette
    ax2.plot(k_list, silhouettes, "ro-", linewidth=2, markersize=8)
    ax2.axvline(x=best_sil_k, color="blue", linestyle="--", label=f"Best Silhouette K = {best_sil_k}")
    ax2.set_xlabel("Số cụm K", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Analysis", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    plt.suptitle("📉 Tìm K tối ưu cho RFM Clustering", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Chọn K: ưu tiên Elbow, tham khảo Silhouette
    best_k = elbow_k
    print(f"\n🏆 K tối ưu: {best_k} (Elbow={elbow_k}, Best Silhouette={best_sil_k})")
    return best_k


def visualize_clusters(model: FoodRecommender):
    """Trực quan hóa các cụm RFM bằng PCA 2D."""

    coords_pca = PCA(n_components=2, random_state=42).fit_transform(model.rfm_scaled)
    labels = model.cluster_labels

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        coords_pca[:, 0], coords_pca[:, 1],
        c=labels, cmap="Set2", alpha=0.6, s=20, edgecolors="k", linewidths=0.3,
    )

    # Vẽ tâm cụm
    pca_for_centers = PCA(n_components=2, random_state=42).fit(model.rfm_scaled)
    centers_pca = pca_for_centers.transform(model.kmeans_model.cluster_centers_)
    plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        c="red", marker="X", s=250, edgecolors="black", linewidths=2,
        label="Tâm cụm", zorder=5,
    )
    for i, (cx, cy) in enumerate(centers_pca):
        plt.annotate(
            f"Cụm {i}", (cx, cy),
            fontsize=11, fontweight="bold",
            ha="center", va="bottom",
            xytext=(0, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
        )

    plt.xlabel(f"PC1 ({pca_for_centers.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"PC2 ({pca_for_centers.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    plt.title("🗺️ Phân cụm Khách hàng RFM (PCA 2D Projection)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig("clusters_pca.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_rfm_boxplot(model: FoodRecommender):
    """Boxplot R/F/M theo từng cụm."""

    rfm_plot = model.rfm.copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, col in enumerate(["Recency", "Frequency", "Monetary"]):
        sns.boxplot(data=rfm_plot, x="Cluster", y=col, ax=axes[idx], palette="Set2")
        axes[idx].set_title(f"{col} theo Cụm", fontsize=13, fontweight="bold")
        axes[idx].set_xlabel("Cụm", fontsize=11)
        axes[idx].set_ylabel(col, fontsize=11)
        axes[idx].grid(True, alpha=0.2)

    plt.suptitle("📊 Phân bố RFM theo Cụm Khách hàng", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("boxplot_rfm.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_cluster_food_heatmap(model: FoodRecommender, top_n: int = 15):
    """Heatmap: Top món ăn phổ biến nhất theo từng cụm RFM."""

    # Xây dựng ma trận: cụm × món (tỷ lệ khách đặt)
    cluster_food_matrix = []
    for c in range(model.n_clusters):
        pop = model.cluster_food_popularity[c]
        n_customers_in_cluster = (model.cluster_labels == c).sum()
        # Tỷ lệ khách trong cụm đã đặt món
        ratio = pop["customer_count"] / n_customers_in_cluster
        cluster_food_matrix.append(ratio)

    df_matrix = pd.DataFrame(cluster_food_matrix).fillna(0)

    # Chọn top_n món có variance cao nhất giữa các cụm
    variance = df_matrix.var(axis=0).sort_values(ascending=False)
    top_foods = variance.head(top_n).index.tolist()
    heatmap_data = df_matrix[top_foods]

    # Đổi tên cột → tên món
    heatmap_data.columns = [
        model.food_id_to_name.get(fid, fid)[:18] for fid in heatmap_data.columns
    ]

    plt.figure(figsize=(16, 6))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".0%", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        xticklabels=True, yticklabels=[f"Cụm {i}" for i in range(model.n_clusters)],
    )
    plt.title(
        f"🔥 Tỷ lệ khách đặt món theo Cụm RFM (Top {top_n} khác biệt nhất)",
        fontsize=13, fontweight="bold", pad=15,
    )
    plt.xlabel("Món ăn", fontsize=11)
    plt.ylabel("Cụm khách hàng", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


def visualize_rfm_radar(model: FoodRecommender):
    """Radar chart thể hiện RFM trung bình của mỗi cụm (chuẩn hóa 0-1)."""

    rfm_means = model.rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Chuẩn hóa 0-1 cho radar
    rfm_norm = rfm_means.copy()
    for col in rfm_norm.columns:
        min_val, max_val = rfm_norm[col].min(), rfm_norm[col].max()
        if max_val > min_val:
            rfm_norm[col] = (rfm_norm[col] - min_val) / (max_val - min_val)
        else:
            rfm_norm[col] = 0.5

    # Đảo Recency (nhỏ = tốt → cần đảo để hiển thị đúng ý nghĩa)
    rfm_norm["Recency"] = 1 - rfm_norm["Recency"]

    categories = ["Recency\n(gần đây)", "Frequency\n(thường xuyên)", "Monetary\n(chi tiêu cao)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Đóng vòng

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, model.n_clusters))

    for c in range(model.n_clusters):
        values = rfm_norm.loc[c].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=f"Cụm {c}", color=colors[c])
        ax.fill(angles, values, alpha=0.15, color=colors[c])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("🎯 Radar RFM theo Cụm", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig("radar_chart.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_cluster_profiles(model: FoodRecommender, top_n: int = 5):
    """In thông tin chi tiết từng cụm: RFM trung bình + top món phổ biến."""

    print("\n" + "=" * 60)
    print("  📋 HỒ SƠ TỪNG CỤM KHÁCH HÀNG (RFM)")
    print("=" * 60)

    for c in range(model.n_clusters):
        cluster_rfm = model.rfm[model.rfm["Cluster"] == c]
        n_members = len(cluster_rfm)

        print(f"\n{'─' * 55}")
        print(f"  🏷️  CỤM {c}  |  {n_members:,} khách ({n_members / len(model.rfm) * 100:.1f}%)")
        print(f"{'─' * 55}")
        print(f"  RFM trung bình:")
        print(f"     Recency  : {cluster_rfm['Recency'].mean():.1f} ngày")
        print(f"     Frequency: {cluster_rfm['Frequency'].mean():.1f} đơn")
        print(f"     Monetary : {cluster_rfm['Monetary'].mean():,.0f} VND")

        # Top món trong cụm
        pop = model.cluster_food_popularity[c].head(top_n)
        print(f"\n  Top {top_n} món yêu thích:")
        print(f"  {'Rank':<5} {'Món ăn':<25} {'Khách đặt':>10} {'Tổng SL':>8}")
        print(f"  {'─'*5} {'─'*25} {'─'*10} {'─'*8}")
        for rank, (fid, row) in enumerate(pop.iterrows(), 1):
            fname = model.food_id_to_name.get(fid, fid)
            print(f"  {rank:<5} {fname:<25} {int(row['customer_count']):>10} {int(row['total_qty']):>8}")


# ============================================================
#  PHẦN 5: CHẠY TOÀN BỘ PIPELINE
# ============================================================

if __name__ == "__main__":

    # ---------- 1. Đọc dữ liệu từ food.csv ----------
    df = load_data()

    print(f"\nMau du lieu (5 dong dau):")
    print(df.head().to_string(index=False))

    # ---------- 2. Tính RFM & Tìm K tối ưu ----------
    rfm = compute_rfm(df)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm.values)

    best_k = find_optimal_k(rfm_scaled, k_range=range(2, 11))

    # ---------- 3. Huấn luyện với K tối ưu ----------
    recommender = FoodRecommender(
        n_clusters=best_k,
        cold_start_threshold=2,
    )
    recommender.fit(df)

    # ---------- 4. Trực quan hóa ----------
    visualize_clusters(recommender)
    visualize_rfm_boxplot(recommender)
    visualize_cluster_food_heatmap(recommender, top_n=15)
    visualize_rfm_radar(recommender)
    print_cluster_profiles(recommender, top_n=5)

    # ---------- 5. Chế độ tương tác: Nhập ID & Gợi ý ----------
    print("\n" + "=" * 60)
    print("  🍜 HỆ THỐNG GỢI Ý MÓN ĂN TƯƠNG TÁC (RFM + K-Means)")
    print("=" * 60)

    print("\n💡 Hướng dẫn:")
    print("   - Nhập ID khách hàng (ví dụ: C0001, C0050, ...)")
    print("   - Hệ thống sẽ gợi ý 5 món ăn phù hợp nhất")
    print("   - Gợi ý dựa trên: lịch sử đặt + cụm RFM + khả năng chi trả")
    print("   - Nhập 'quit' để thoát\n")

    while True:
        try:
            customer_id = input("Nhập ID khách hàng (hoặc 'quit' để thoát): ").strip()

            if customer_id.lower() == "quit":
                print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
                break

            if not customer_id:
                print("⚠️  Vui lòng nhập ID khách hàng!")
                continue

            print(f"\n{'━' * 70}")
            rec = recommender.recommend(customer_id, top_k=5)
            print(rec.to_string(index=False))
            print(f"{'━' * 70}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Đã dừng chương trình!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            continue
