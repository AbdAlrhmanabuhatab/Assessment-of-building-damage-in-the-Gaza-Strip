import os
import sys

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

from preprocess import load_and_preprocess_image, CLASS_NAMES
from gradcam import get_gradcam_heatmap, overlay_gradcam

# Page setup
st.set_page_config(
    page_title="نظام تصنيف أضرار مباني غزة | AI Damage Detection",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .main-header {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 25px;
        border: 1px solid #334155;
    }
    .badge-demolish {
        background-color: #ef4444;
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: inline-block;
    }
    .badge-repair {
        background-color: #f59e0b;
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: inline-block;
    }
    .badge-safe {
        background-color: #10b981;
        color: white;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 20px;
        display: inline-block;
    }
    .card {
        background-color: #1e293b;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_ai_model(model_path):
    return tf.keras.models.load_model(model_path)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 28px;">🏢 نظام الذكاء الاصطناعي لتقييم أضرار المباني في قطاع غزة</h1>
    <p style="margin:6px 0 0 0; color: #94a3b8; font-size: 15px;">
        منظومة تحليل الصور الإنشائية باستخدام شبكات التعلم العميق (MobileNetV2) والذكاء الاصطناعي القابل للتفسير (Grad-CAM)
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ إعدادات النموذج والبيانات")
    
    # Model Selection
    available_models = {}
    if os.path.exists("building_damage_model_v2.keras"):
        available_models["النموذج المطور V2 (Keras - موصى به)"] = ("building_damage_model_v2.keras", "mobilenet")
    if os.path.exists("building_damage_model_finetuned.h5"):
        available_models["النموذج السابق المحسّن (Finetuned .h5)"] = ("building_damage_model_finetuned.h5", "div255")
    if os.path.exists("building_damage_model.h5"):
        available_models["النموذج الأساسي الأولي (Base .h5)"] = ("building_damage_model.h5", "div255")

    chosen_name = st.selectbox("اختر إصدار النموذج:", list(available_models.keys()))
    model_path, norm_mode = available_models[chosen_name]

    st.info(f"📁 **الملف:** `{model_path}`\n\n🔬 **نمط التطبيع:** `{norm_mode}`")

    st.markdown("---")
    st.subheader("🔍 إعدادات Grad-CAM")
    show_gradcam = st.checkbox("تفعيل خريطة التفسير البصري (Grad-CAM)", value=True)
    alpha = st.slider("شفافية خريطة التنشيط الحراري (Overlay Opacity):", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    st.markdown("---")
    st.markdown("📌 **فئات التصنيف:**")
    st.markdown("🔴 **Demolish**: ضرر كلي / هدم / انهيار إنشائي")
    st.markdown("🟡 **Repair**: ضرر جزئي / قابل للترميم")
    st.markdown("🟢 **Safe**: سليم / آمن")

# Load selected model
try:
    model = load_ai_model(model_path)
except Exception as e:
    st.error(f"خطأ أثناء تحميل النموذج: {e}")
    st.stop()

# Image selection or upload
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📤 اختيار صورة المبنى للفحص")
    source_choice = st.radio("مصدر الصورة:", ["رفع صورة من الجهاز", "اختيار صورة تجريبية من النظام"], horizontal=True)
    
    input_image_bytes = None
    input_image_name = ""

    if source_choice == "رفع صورة من الجهاز":
        uploaded_file = st.file_uploader("اختر صورة (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image_bytes = uploaded_file.read()
            input_image_name = uploaded_file.name
    else:
        # Sample images
        sample_options = ["test_image.jpg"]
        test_dir = os.path.join("dataset_split", "test")
        if os.path.exists(test_dir):
            for c in ["demolish", "repair", "safe"]:
                c_dir = os.path.join(test_dir, c)
                if os.path.exists(c_dir):
                    sample_options.extend([os.path.join(c_dir, f) for f in os.listdir(c_dir)[:2]])
        
        selected_sample = st.selectbox("اختر من العينات المتاحة:", sample_options)
        if selected_sample and os.path.exists(selected_sample):
            with open(selected_sample, "rb") as f:
                input_image_bytes = f.read()
            input_image_name = os.path.basename(selected_sample)

    if input_image_bytes is not None:
        pil_img = Image.open(io := sys.modules['io'].BytesIO(input_image_bytes))
        st.image(pil_img, caption=f"الصورة الأصلية: {input_image_name}", use_container_width=True)

# Analysis and Results
with col_right:
    st.subheader("📊 نتائج التحليل الإنشائي")
    
    if input_image_bytes is None:
        st.info("👆 يرجى رفع صورة أو اختيار عينة من العمود المقابل للبدء بالتحليل الميداني.")
    else:
        with st.spinner("جاري معالجة الصورة وتحليل الميزات الإنشائية..."):
            img_input, rgb_orig = load_and_preprocess_image(input_image_bytes, mode=norm_mode)
            preds = model.predict(img_input, verbose=0)[0]
            top_idx = int(np.argmax(preds))
            top_label = CLASS_NAMES[top_idx]
            confidence = float(preds[top_idx]) * 100.0

        # Class Badges
        if top_label == "Demolish":
            badge_html = f'<div class="badge-demolish">⚠️ قرار إخلاء فوري: ضرر كلي / هدم ({top_label})</div>'
            advisory = "⚠️ **خطر إنشائي داهم:** المبنى يعاني من انهيارات في الأعمدة أو السقف. يُحظر الدخول تماماً ويجب إخلاء محيط المبنى فوراً."
        elif top_label == "Repair":
            badge_html = f'<div class="badge-repair">🛠️ قرار معاينة: ضرر جزئي / قابل للترميم ({top_label})</div>'
            advisory = "🛠️ **ضرر إنشائي متوسط:** المبنى تضرر جزئياً ولكن الهيكل الأساسي متماسك. يتطلب كشفاً ميدانياً من مهندس نقابي قبل إعادة الإشغال."
        else:
            badge_html = f'<div class="badge-safe">✅ حالة آمنة: لا توجد أضرار حرجة ({top_label})</div>'
            advisory = "✅ **هيكل آمن ظاهرياً:** لا تظهر علامات تصدع حرج أو انهيار من هذه الزاوية. يُنصح بالتأكد من عدم وجود تشققات داخلية مخفية."

        st.markdown(badge_html, unsafe_allow_html=True)
        st.metric(label="درجة ثقة النموذج في القرار", value=f"{confidence:.2f} %")

        st.markdown("---")
        st.write("📈 **توزيع احتمالات درجات الضرر:**")
        
        prob_safe = float(preds[0]) * 100
        prob_repair = float(preds[1]) * 100
        prob_demolish = float(preds[2]) * 100

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("آمن (Safe)", f"{prob_safe:.1f}%")
        col_b.metric("ترميم (Repair)", f"{prob_repair:.1f}%")
        col_c.metric("هدم (Demolish)", f"{prob_demolish:.1f}%")

        # Progress bars
        st.write("Safe:")
        st.progress(prob_safe / 100.0)
        st.write("Repair:")
        st.progress(prob_repair / 100.0)
        st.write("Demolish:")
        st.progress(prob_demolish / 100.0)

        st.info(advisory)

# Grad-CAM Section below
if input_image_bytes is not None and show_gradcam:
    st.markdown("---")
    st.subheader("🔬 التفسير البصري للقرار (Explainable AI - Grad-CAM)")
    st.caption("توضح الخريطة الحرارية المناطق التي ركز عليها الذكاء الاصطناعي لاستنتاج درجة الضرر (اللون الأحمر/الأصفر = تركيز عالٍ، الأزرق = تركيز منخفض).")

    with st.spinner("جاري توليد خريطة Grad-CAM..."):
        try:
            heatmap = get_gradcam_heatmap(img_input, model, pred_index=top_idx)
            overlay = overlay_gradcam(rgb_orig, heatmap, alpha=alpha)

            g_col1, g_col2, g_col3 = st.columns(3)
            with g_col1:
                st.image(rgb_orig, caption="1. الصورة الأصلية (RGB)", use_container_width=True)
            with g_col2:
                # Plot heatmap with matplotlib to show colorbar
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(heatmap, cmap="jet")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
            with g_col3:
                st.image(overlay, caption=f"3. تراكب التركيز الإنشائي ({top_label})", use_container_width=True)

        except Exception as e:
            st.warning(f"تعذر حساب Grad-CAM لهذا النموذج أو الصورة: {e}")
