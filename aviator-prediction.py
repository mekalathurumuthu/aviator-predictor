import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Aviator Pattern Analyzer", layout="centered")
st.title("✈️ Aviator Pattern Analyzer & Next-Value Predictor")

st.write("""
Enter one multiplier value at a time.  
The app will keep track of your inputs, show a live chart,  
and calculate average and basic predictions.
""")

# -------------------------------
# Session state initialization
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# Input Section
# -------------------------------
st.subheader("Enter a new multiplier value")
new_value = st.number_input("Multiplier", min_value=0.0, step=0.1, format="%.2f")

col1, col2 = st.columns(2)
with col1:
    if st.button("➕ Add Value"):
        st.session_state.history.append(new_value)
        st.success(f"Added value: {new_value}")
with col2:
    if st.button("🗑️ Clear All"):
        st.session_state.history.clear()
        st.warning("All data cleared!")

# -------------------------------
# Display History
# -------------------------------
if len(st.session_state.history) > 0:
    st.subheader("📊 Multiplier History")
    df = pd.DataFrame(st.session_state.history, columns=["Multiplier"])
    st.dataframe(df, use_container_width=True)

    # -------------------------------
    # Chart Visualization
    # -------------------------------
    st.subheader("📈 Multiplier Trend")
    fig, ax = plt.subplots()
    ax.plot(df["Multiplier"], marker='o', linestyle='-', linewidth=2)
    ax.set_xlabel("Entry Number")
    ax.set_ylabel("Multiplier Value")
    ax.set_title("Trend of Entered Multipliers")
    st.pyplot(fig)

    # -------------------------------
    # Simple Prediction Logic
    # -------------------------------
    avg_val = df["Multiplier"].mean()
    last_val = df["Multiplier"].iloc[-1]
    predicted_next = round((avg_val + last_val) / 2, 2)

    st.subheader("🔮 Prediction Summary")
    st.info(f"Average Value so far: **{avg_val:.2f}**")
    st.success(f"Predicted Next Multiplier: **{predicted_next:.2f}**")
else:
    st.warning("No values entered yet. Please add one above.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Mekalathuru Muthu | Powered by Streamlit")
