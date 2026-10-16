import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("queue_waiting_model.pkl")
model_features = joblib.load("model_features.pkl")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Queue Management",
    page_icon="⏳",
    layout="centered"
)

st.title("⏳ Smart Queue Management")
st.subheader("Waiting Time Prediction System")

st.write(
    "Predict waiting time for different real-world queue environments."
)


# -----------------------------
# Select Individual Use Case
# -----------------------------
use_case = st.selectbox(
    "Select Use Case",
    ["🏦 Bank", "🏥 Hospital", "🏢 Office"]
)

# Convert selection to model queue type
if use_case == "🏦 Bank":
    queue_type = "Bank"
elif use_case == "🏥 Hospital":
    queue_type = "Hospital"
else:
    queue_type = "Office"


st.markdown("---")

st.subheader(f"{use_case} Queue Details")


# -----------------------------
# Inputs
# -----------------------------
number_of_people = st.number_input(
    "Number of People",
    min_value=1,
    max_value=500,
    value=30
)

counters_open = st.number_input(
    "Counters Open",
    min_value=1,
    max_value=20,
    value=3
)

avg_service_time = st.number_input(
    "Average Service Time (minutes)",
    min_value=1,
    max_value=60,
    value=10
)

time_of_day = st.slider(
    "Time of Day",
    min_value=0,
    max_value=23,
    value=12
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Waiting Time"):

    # Prepare input according to trained model
    input_data = pd.DataFrame({
        "Number_of_People": [number_of_people],
        "Counters_Open": [counters_open],
        "Avg_Service_Time": [avg_service_time],
        "Time_of_Day": [time_of_day],
        "Queue_Type_Hospital": [
            1 if queue_type == "Hospital" else 0
        ],
        "Queue_Type_Office": [
            1 if queue_type == "Office" else 0
        ]
    })

    # Prediction
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)

    # -----------------------------
    # Display Prediction
    # -----------------------------
    st.success(
        f"Estimated Waiting Time: {prediction:.2f} minutes"
    )

    st.markdown("---")

    # -----------------------------
    # Use-Case Specific Message
    # -----------------------------
    if queue_type == "Bank":

        st.subheader("🏦 Bank - While You Wait")

        if prediction <= 15:
            suggestion = (
                "You have a short waiting time. "
                "You can check your banking documents, "
                "messages, or make a quick phone call."
            )
        elif prediction <= 30:
            suggestion = (
                "You have some waiting time. "
                "You can check emails, organize your documents, "
                "or have a coffee."
            )
        else:
            suggestion = (
                "You have considerable waiting time. "
                "You can complete another small task, "
                "check your finances, or take a short break."
            )

    elif queue_type == "Hospital":

        st.subheader("🏥 Hospital - While You Wait")

        if prediction <= 15:
            suggestion = (
                "You have a short waiting time. "
                "You can complete registration or check "
                "your appointment details."
            )
        elif prediction <= 30:
            suggestion = (
                "You have some waiting time. "
                "You can complete pending forms, "
                "read, or take a short break."
            )
        else:
            suggestion = (
                "You have considerable waiting time. "
                "You can complete pending paperwork, "
                "have refreshments, or relax while waiting."
            )

    else:

        st.subheader("🏢 Office - While You Wait")

        if prediction <= 15:
            suggestion = (
                "You have a short waiting time. "
                "You can check messages, emails, or make a quick call."
            )
        elif prediction <= 30:
            suggestion = (
                "You have some waiting time. "
                "You can check emails, organize your work, "
                "or complete a small task."
            )
        else:
            suggestion = (
                "You have considerable waiting time. "
                "You can complete another work task, "
                "attend a call, or take a short break."
            )

    st.info("💡 " + suggestion)

    # -----------------------------
    # Individual Use-Case Graph
    # -----------------------------
    st.markdown("---")
    st.subheader(f"{use_case} Waiting Time Graph")

    # Different numbers of people
    people_values = list(range(10, 201, 10))

    predicted_waiting_times = []

    for people in people_values:

        graph_input = pd.DataFrame({
            "Number_of_People": [people],
            "Counters_Open": [counters_open],
            "Avg_Service_Time": [avg_service_time],
            "Time_of_Day": [time_of_day],
            "Queue_Type_Hospital": [
                1 if queue_type == "Hospital" else 0
            ],
            "Queue_Type_Office": [
                1 if queue_type == "Office" else 0
            ]
        })

        graph_prediction = model.predict(graph_input)[0]
        predicted_waiting_times.append(max(0, graph_prediction))

    # Create graph
    fig, ax = plt.subplots()

    ax.plot(
        people_values,
        predicted_waiting_times,
        marker="o"
    )

    ax.set_xlabel("Number of People")
    ax.set_ylabel("Predicted Waiting Time (minutes)")
    ax.set_title(f"{queue_type} - People vs Waiting Time")

    ax.grid(True)

    st.pyplot(fig)

    st.caption(
        "The graph shows how the predicted waiting time changes "
        "as the number of people in the queue increases."
    )