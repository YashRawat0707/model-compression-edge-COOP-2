import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Compression Demo", layout="centered")


st.title("🚀 Model Compression for Edge Devices")

st.write("Compare different compression techniques and their impact on model performance.")


data = {
    "Model": ["Original", "Pruned", "Quantized", "Hybrid"],
    "Size (MB)": [2.34, 2.34, 0.64, 0.60],
    "Accuracy (%)": [69.95, 67.02, 69.92, 66.50]
}

df = pd.DataFrame(data)


df["Efficiency"] = df["Accuracy (%)"] / df["Size (MB)"]

best_model = df.sort_values(by="Efficiency", ascending=False).iloc[0]


selected_model = st.selectbox("Select Model", df["Model"])

model_data = df[df["Model"] == selected_model].iloc[0]


st.subheader("📊 Selected Model Details")

col1, col2 = st.columns(2)

col1.metric("Size (MB)", model_data["Size (MB)"])
col2.metric("Accuracy (%)", model_data["Accuracy (%)"])


st.subheader("🏆 Best Model Recommendation")

st.success(f"Best Model: {best_model['Model']} "
           f"(Accuracy: {best_model['Accuracy (%)']}%, Size: {best_model['Size (MB)']} MB)")


st.subheader("📈 Comparison Table")
st.dataframe(df, use_container_width=True)


st.subheader("📊 Visual Comparison")

st.write("### Accuracy Comparison")
st.bar_chart(df.set_index("Model")["Accuracy (%)"])

st.write("### Size Comparison")
st.bar_chart(df.set_index("Model")["Size (MB)"])


st.write("### Accuracy vs Size Trade-off")

st.scatter_chart(
    df,
    x="Size (MB)",
    y="Accuracy (%)"
)


st.subheader("🧠 Insights")

st.info("""
- Quantization provides the best size reduction with almost no accuracy loss.
- Pruning reduces model complexity but does not reduce file size significantly.
- Hybrid approach combines both but may slightly reduce accuracy.
- Trade-off between size and accuracy is key for edge deployment.
""")