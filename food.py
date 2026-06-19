"""
Hệ thống Phân tích Khách hàng bằng RFM + K-Means
==================================================
Pipeline:
  1. Đọc dữ liệu đơn hàng từ food.csv (sinh bởi gen_data.py)
  2. Tính RFM: Recency, Frequency, Monetary, AOV
  3. Loại bỏ Outlier bằng IQR
  4. Chuẩn hóa (StandardScaler)
  5. Tìm K tối ưu (Elbow + Silhouette)
  6. Chạy K-Means phân cụm
  7. Dán nhãn kinh doanh (VIP / Phổ thông / Giá trị thấp)
  8. Trực quan hóa + Báo cáo
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import cm
import warnings
import os

warnings.filterwarnings("ignore")


# ============================================================
#  PHẦN 1: ĐỌC DỮ LIỆU TỪ food.csv
# ============================================================

def load_data(csv_path: str = None) -> pd.DataFrame:
    """
    Đọc dữ liệu đơn hàng từ file food.csv.

    File food.csv được sinh bởi gen_data.py với cấu trúc:
        InvoiceNo, InvoiceDate, CustomerID, FoodID, FoodName, Quantity, UnitPrice
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "food.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {csv_path}\n"
            f"Hãy chạy 'python gen_data.py' trước để sinh dữ liệu."
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["InvoiceDate"])

    print(f"Đã đọc {len(df):,} dòng dữ liệu từ {os.path.basename(csv_path)}")
    print(f"   Số khách hàng : {df['CustomerID'].nunique():,}")
    print(f"   Số đơn hàng   : {df['InvoiceNo'].nunique():,}")
    print(f"   Số món ăn     : {df['FoodID'].nunique()}")
    print(f"   Khoảng thời gian: {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}")
    return df


# ============================================================
#  PHẦN 2: TÍNH RFM + TIỀN XỬ LÝ
# ============================================================

def IQR_filter(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Loại bỏ outlier bằng IQR (giống kh/main.py)."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính RFM cho từng khách hàng.

    Returns:
        DataFrame với index=CustomerID, columns=[Recency, Frequency, Monetary, AOV]
    """
    # Tính TotalPrice cho mỗi dòng
    df = df.copy()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Ngày phân tích = ngày max + 1
    analysis_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Tính RFM
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (analysis_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )

    # AOV = Average Order Value
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]

    # Chỉ giữ khách có Monetary > 0
    rfm = rfm[rfm["Monetary"] > 0]

    print(f"\n📊 Bảng RFM: {len(rfm):,} khách hàng")
    print(f"   Recency  : {rfm['Recency'].min():.0f} → {rfm['Recency'].max():.0f} ngày")
    print(f"   Frequency: {rfm['Frequency'].min():.0f} → {rfm['Frequency'].max():.0f} đơn")
    print(f"   Monetary : {rfm['Monetary'].min():,.0f}đ → {rfm['Monetary'].max():,.0f}đ")
    print(f"   AOV      : {rfm['AOV'].min():,.0f}đ → {rfm['AOV'].max():,.0f}đ")

    return rfm


# ============================================================
#  PHẦN 3: PIPELINE PHÂN CỤM K-MEANS
# ============================================================

class FoodCustomerAnalyzer:
    """
    Hệ thống phân tích khách hàng đặt đồ ăn bằng RFM + K-Means.

    Pipeline:
        1. Tính RFM → Loại outlier
        2. Chuẩn hóa (StandardScaler)
        3. Tìm K tối ưu (Elbow + Silhouette)
        4. Chạy K-Means phân cụm
        5. Dán nhãn kinh doanh
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state

        # Kết quả sau khi fit
        self.rfm = None
        self.rfm_clean = None           # Sau loại outlier
        self.scaler = None
        self.rfm_scaled = None
        self.kmeans_model = None
        self.cluster_labels = None
        self.df_clustered = None        # RFM + Segment
        self.profiles = None            # Profile trung bình từng cụm

        # Metadata cho visualization
        self.k_range = None
        self.inertia_list = None
        self.silhouette_list = None

    def fit(self, df: pd.DataFrame):
        """Huấn luyện mô hình trên toàn bộ dữ liệu."""

        print("\n" + "=" * 60)
        print("  🔧 BẮT ĐẦU PHÂN TÍCH KHÁCH HÀNG RFM + K-MEANS")
        print("=" * 60)

        # --- Bước 1: Tính RFM ---
        self.rfm = build_rfm(df)

        # --- Bước 2: Loại bỏ Outlier ---
        self.rfm_clean = self.rfm.copy()
        n_before = len(self.rfm_clean)
        for col in ["Recency", "Frequency", "Monetary", "AOV"]:
            self.rfm_clean = IQR_filter(self.rfm_clean, col)
        n_after = len(self.rfm_clean)

        print(f"\n🧹 Bước 2 – Loại Outlier (IQR)")
        print(f"   Trước: {n_before:,} → Sau: {n_after:,} khách (loại {n_before - n_after:,})")

        # --- Bước 3: Chuẩn hóa ---
        self.scaler = StandardScaler()
        features = self.rfm_clean[["Recency", "Frequency", "Monetary", "AOV"]]
        self.rfm_scaled = self.scaler.fit_transform(features)

        print(f"\n📐 Bước 3 – Chuẩn hóa (StandardScaler)")
        print(f"   Kích thước ma trận: {self.rfm_scaled.shape[0]:,} × {self.rfm_scaled.shape[1]}")

        # --- Bước 4: Tìm K tối ưu ---
        self.k_range = range(2, 11)
        self.inertia_list = []
        self.silhouette_list = []

        print(f"\n🔍 Bước 4 – Tìm K tối ưu (Elbow + Silhouette)")
        for k in self.k_range:
            km = KMeans(
                n_clusters=k, init="k-means++",
                n_init=10, max_iter=300,
                random_state=self.random_state,
            )
            km.fit(self.rfm_scaled)
            self.inertia_list.append(km.inertia_)
            sil = silhouette_score(self.rfm_scaled, km.labels_)
            self.silhouette_list.append(sil)
            marker = " ← đang dùng" if k == self.n_clusters else ""
            print(f"   K={k}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.4f}{marker}")

        # --- Bước 5: Chạy K-Means ---
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )
        self.cluster_labels = self.kmeans_model.fit_predict(self.rfm_scaled)

        # --- Bước 6: Dán nhãn kinh doanh ---
        self.df_clustered = self.rfm_clean.copy()
        self.df_clustered["Cluster"] = self.cluster_labels

        # Dán nhãn dựa trên Monetary trung bình
        cluster_monetary = (
            self.df_clustered.groupby("Cluster")["Monetary"]
            .mean()
            .sort_values(ascending=False)
        )

        segment_names = ["Khách VIP", "Khách Phổ thông", "Khách Giá trị thấp"]
        segment_map = {}
        for idx, cluster_id in enumerate(cluster_monetary.index):
            if idx < len(segment_names):
                segment_map[cluster_id] = segment_names[idx]
            else:
                segment_map[cluster_id] = f"Nhóm {cluster_id}"

        self.df_clustered["Segment"] = self.df_clustered["Cluster"].map(segment_map)

        # --- Bước 7: Tạo bảng profile ---
        self.profiles = (
            self.df_clustered
            .groupby("Segment")[["Recency", "Frequency", "Monetary", "AOV"]]
            .mean()
        )
        self.profiles["customer_n"] = self.df_clustered.groupby("Segment").size()
        self.profiles = self.profiles.sort_values("Monetary", ascending=False)

        print(f"\n🎯 Bước 5-6 – K-Means Clustering (K={self.n_clusters})")
        for seg_name, row in self.profiles.iterrows():
            n = int(row["customer_n"])
            pct = n / len(self.df_clustered) * 100
            print(f"   • {seg_name}: {n:,} khách ({pct:.1f}%)")

        print("\n✅ PHÂN TÍCH HOÀN TẤT!")
        return self

    def generate_report(self):
        """In báo cáo phân tích + đề xuất marketing."""

        print("\n" + "=" * 60)
        print("  📋 BÁO CÁO PHÂN CỤM KHÁCH HÀNG")
        print("=" * 60)
        print(f"Tổng khách hàng phân tích: {len(self.df_clustered):,}")

        for seg_name, row in self.profiles.iterrows():
            n = int(row["customer_n"])
            pct = n / len(self.df_clustered) * 100

            print(f"\n{'─' * 55}")
            print(f"  🏷️  {seg_name}  |  {n:,} khách ({pct:.1f}%)")
            print(f"{'─' * 55}")
            print(f"  Recency TB   : {row['Recency']:,.1f} ngày")
            print(f"  Frequency TB : {row['Frequency']:,.1f} đơn hàng")
            print(f"  Monetary TB  : {row['Monetary']:,.0f}đ")
            print(f"  AOV TB       : {row['AOV']:,.0f}đ")

            # Đề xuất marketing
            if "VIP" in seg_name:
                print(f"  📌 Đề xuất: Chương trình loyalty, ưu đãi độc quyền,")
                print(f"              combo cao cấp, phục vụ ưu tiên")
            elif "Phổ thông" in seg_name:
                print(f"  📌 Đề xuất: Upsell món đắt hơn, tích điểm thưởng,")
                print(f"              combo tiết kiệm, voucher khi đặt thêm")
            elif "Giá trị thấp" in seg_name:
                print(f"  📌 Đề xuất: Flash sale, mã giảm giá kích hoạt lại,")
                print(f"              push notification nhắc nhở, freeship")

        print(f"\n{'=' * 60}")
        print("Biểu đồ đã lưu trong thư mục hiện tại.")


# ============================================================
#  PHẦN 4: TRỰC QUAN HÓA
# ============================================================

def plot_elbow_silhouette(analyzer: FoodCustomerAnalyzer):
    """Vẽ biểu đồ Elbow + Silhouette Score."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    k_list = list(analyzer.k_range)

    # Elbow
    ax1.plot(k_list, analyzer.inertia_list, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax1.fill_between(k_list, analyzer.inertia_list, alpha=0.1, color="#2196F3")
    ax1.axvline(x=analyzer.n_clusters, color="green", linestyle="--",
                label=f"K = {analyzer.n_clusters}")
    ax1.set_xlabel("Số cụm K", fontsize=12)
    ax1.set_ylabel("Inertia (SSE)", fontsize=12)
    ax1.set_title("📉 Elbow Method", fontsize=14, fontweight="bold")
    ax1.set_xticks(k_list)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Silhouette
    ax2.plot(k_list, analyzer.silhouette_list, "s-", color="#E91E63", linewidth=2, markersize=8)
    ax2.fill_between(k_list, analyzer.silhouette_list, alpha=0.1, color="#E91E63")
    ax2.axvline(x=analyzer.n_clusters, color="green", linestyle="--",
                label=f"K = {analyzer.n_clusters}")
    ax2.set_xlabel("Số cụm K", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("📊 Silhouette Score", fontsize=14, fontweight="bold")
    ax2.set_xticks(k_list)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Đánh dấu K tốt nhất (Silhouette cao nhất)
    best_k_idx = np.argmax(analyzer.silhouette_list)
    best_k = k_list[best_k_idx]
    best_score = analyzer.silhouette_list[best_k_idx]
    ax2.annotate(
        f"Best K={best_k}\n({best_score:.3f})",
        xy=(best_k, best_score),
        xytext=(best_k + 0.8, best_score),
        fontsize=10, fontweight="bold", color="#E91E63",
        arrowprops=dict(arrowstyle="->", color="#E91E63"),
    )

    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Đã lưu: elbow_silhouette.png")


def plot_pca_clusters(analyzer: FoodCustomerAnalyzer):
    """Trực quan hóa các cụm bằng PCA 2D."""

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(analyzer.rfm_scaled)
    centers_pca = pca.transform(analyzer.kmeans_model.cluster_centers_)

    plt.figure(figsize=(12, 8))

    # Lấy nhãn segment cho mỗi điểm
    segments = analyzer.df_clustered["Segment"].values
    unique_segments = analyzer.profiles.index.tolist()

    palette = {
        "Khách VIP": "#FFD700",
        "Khách Phổ thông": "#2196F3",
        "Khách Giá trị thấp": "#9E9E9E",
    }

    for seg in unique_segments:
        mask = segments == seg
        color = palette.get(seg, "#888888")
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, alpha=0.6, s=30, edgecolors="white",
            linewidths=0.3, label=seg,
        )

    # Vẽ tâm cụm
    plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        c="red", marker="X", s=250, edgecolors="black",
        linewidths=2, label="Tâm cụm", zorder=5,
    )

    # Nhãn tâm cụm
    cluster_to_seg = (
        analyzer.df_clustered
        .groupby("Cluster")["Segment"]
        .first()
        .to_dict()
    )
    for i, (cx, cy) in enumerate(centers_pca):
        seg_label = cluster_to_seg.get(i, f"Cụm {i}")
        plt.annotate(
            seg_label, (cx, cy),
            fontsize=10, fontweight="bold",
            ha="center", va="bottom",
            xytext=(0, 12), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
        )

    explained = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=12)
    plt.title("🗺️ Phân cụm Khách hàng (PCA 2D Projection)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("clusters_pca.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Đã lưu: clusters_pca.png")


def plot_silhouette_detail(analyzer: FoodCustomerAnalyzer):
    """Silhouette Plot chi tiết — biểu đồ đặc trưng của K-Means."""

    fig, ax = plt.subplots(figsize=(10, 7))

    sil_avg = silhouette_score(analyzer.rfm_scaled, analyzer.cluster_labels)
    sample_sil = silhouette_samples(analyzer.rfm_scaled, analyzer.cluster_labels)

    y_lower = 10
    cmap_obj = cm.get_cmap("Spectral")
    colors = [cmap_obj(float(i) / analyzer.n_clusters) for i in range(analyzer.n_clusters)]

    for i in range(analyzer.n_clusters):
        ith_values = sample_sil[analyzer.cluster_labels == i]
        ith_values.sort()

        size_i = ith_values.shape[0]
        y_upper = y_lower + size_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, ith_values,
            facecolor=colors[i], edgecolor=colors[i], alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i),
                fontsize=11, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=1.5,
               label=f"Silhouette TB = {sil_avg:.3f}")
    ax.set_xlabel("Giá trị Silhouette", fontsize=12)
    ax.set_ylabel("Cụm (Cluster)", fontsize=12)
    ax.set_title(
        f"📊 Silhouette Plot (K={analyzer.n_clusters})",
        fontsize=14, fontweight="bold",
    )
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=11)
    plt.tight_layout()
    plt.savefig("silhouette_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Đã lưu: silhouette_plot.png")


def plot_centroid_heatmap(analyzer: FoodCustomerAnalyzer):
    """Heatmap giá trị centroid (đã chuẩn hóa)."""

    feature_names = ["Recency", "Frequency", "Monetary", "AOV"]
    cluster_to_seg = (
        analyzer.df_clustered
        .groupby("Cluster")["Segment"]
        .first()
        .to_dict()
    )
    cluster_labels = [cluster_to_seg.get(i, f"Cụm {i}") for i in range(analyzer.n_clusters)]

    plt.figure(figsize=(10, 4))
    sns.heatmap(
        analyzer.kmeans_model.cluster_centers_,
        annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=feature_names,
        yticklabels=cluster_labels,
        linewidths=0.5,
    )
    plt.title("🔥 Giá trị Centroid (chuẩn hóa)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("centroid_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Đã lưu: centroid_heatmap.png")


def plot_rfm_distributions(analyzer: FoodCustomerAnalyzer):
    """Boxplot phân phối RFM theo từng cụm."""

    features = ["Recency", "Frequency", "Monetary", "AOV"]
    n = len(features)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    palette = {
        "Khách VIP": "#FFD700",
        "Khách Phổ thông": "#2196F3",
        "Khách Giá trị thấp": "#9E9E9E",
    }

    for ax, feat in zip(axes, features):
        sns.boxplot(
            data=analyzer.df_clustered, x="Segment", y=feat, ax=ax,
            palette=palette, width=0.6,
        )
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "📦 Phân phối RFM theo Cụm khách hàng",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("boxplot_rfm.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Đã lưu: boxplot_rfm.png")


def plot_radar_chart(analyzer: FoodCustomerAnalyzer):
    """Radar chart so sánh profile các cụm."""

    features = ["Recency", "Frequency", "Monetary", "AOV"]
    profiles = analyzer.profiles[features].copy()

    # Chuẩn hóa min-max cho radar
    for col in features:
        col_min, col_max = profiles[col].min(), profiles[col].max()
        if col_max > col_min:
            profiles[col] = (profiles[col] - col_min) / (col_max - col_min)
        else:
            profiles[col] = 0.5

    n_features = len(features)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    palette = ["#FFD700", "#2196F3", "#9E9E9E", "#FF5722", "#4CAF50"]
    for idx, (seg_name, row) in enumerate(profiles.iterrows()):
        vals = row.tolist()
        vals += vals[:1]
        color = palette[idx % len(palette)]
        ax.plot(angles, vals, "o-", linewidth=2, label=seg_name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=11)
    ax.set_title("🕸️ Radar Chart — So sánh Profile cụm", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig("radar_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Đã lưu: radar_chart.png")


def plot_segment_pie(analyzer: FoodCustomerAnalyzer):
    """Biểu đồ tròn tỷ lệ khách hàng theo cụm."""

    counts = analyzer.df_clustered["Segment"].value_counts()

    palette = {
        "Khách VIP": "#FFD700",
        "Khách Phổ thông": "#2196F3",
        "Khách Giá trị thấp": "#9E9E9E",
    }
    colors = [palette.get(seg, "#888") for seg in counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor="white"),
    )

    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")

    ax.set_title(
        "🥧 Tỷ lệ Khách hàng theo Phân khúc",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.tight_layout()
    plt.savefig("segment_pie.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Đã lưu: segment_pie.png")


# ============================================================
#  PHẦN 5: CHẠY TOÀN BỘ PIPELINE
# ============================================================

if __name__ == "__main__":

    # ---------- 1. Đọc dữ liệu từ food.csv ----------
    df = load_data()

    print(f"\nMẫu dữ liệu (5 dòng đầu):")
    print(df.head().to_string(index=False))

    # ---------- 2. Phân tích RFM + K-Means ----------
    analyzer = FoodCustomerAnalyzer(n_clusters=3)
    analyzer.fit(df)

    # ---------- 3. Trực quan hóa ----------
    plot_elbow_silhouette(analyzer)
    plot_pca_clusters(analyzer)
    plot_silhouette_detail(analyzer)
    plot_centroid_heatmap(analyzer)
    plot_rfm_distributions(analyzer)
    plot_radar_chart(analyzer)
    plot_segment_pie(analyzer)

    # ---------- 4. Báo cáo ----------
    analyzer.generate_report()

    # ---------- 5. Lưu kết quả ----------
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "customer_rfm_clustered.csv",
    )
    analyzer.df_clustered.to_csv(output_path, encoding="utf-8-sig")
    print(f"\n💾 Đã lưu kết quả phân cụm: {output_path}")
