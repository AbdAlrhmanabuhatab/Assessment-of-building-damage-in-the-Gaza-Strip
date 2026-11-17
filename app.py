import tkinter as tk
from tkinter import filedialog
import requests
import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

API_URL = "http://127.0.0.1:8000/predict"

def select_and_analyze():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="اختر صورة مبنى للفحص",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    
    if not file_path:
        print("[INFO] لم يتم اختيار صورة.")
        return

    print(f"\n[INFO] جاري إرسال الصورة للسيرفر: {os.path.basename(file_path)}")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})
        
        if response.status_code != 200:
            print(f"[ERROR] فشل التحليل: {response.text}")
            return

        data = response.json()
        
        print("\n" + "=" * 50)
        print("         تقرير تقييم الأضرار الإنشائية")
        print("=" * 50)
        print(f"  التصنيف النهائي : {data['prediction']}")
        print(f"  نسبة الثقة     : {data['confidence'] * 100:.2f}%")
        print("-" * 50)
        print("  تفصيل الاحتمالات لكل فئة:")
        for cls_name, prob in data.get("probabilities", {}).items():
            bar = "#" * int(prob * 25)
            print(f"    - {cls_name:<10}: {prob * 100:6.2f}% | {bar}")
        print("-" * 50)
        print(f"  التوصية: {data.get('recommendation', 'لا توجد توصية')}")
        print("=" * 50 + "\n")

    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] تعذر الاتصال بـ API على {API_URL}.")
        print("يرجى التأكد من تشغيل الخادم أولاً عبر الأمر: py api.py\n")
    except Exception as e:
        print(f"[ERROR] حدث خطأ غير متوقع: {e}")

if __name__ == "__main__":
    print("\n--- نظام فحص أضرار المباني في قطاع غزة ---")
    print("اضغط Enter لفتح نافذة اختيار الصورة من جهازك...")
    input()
    select_and_analyze()
