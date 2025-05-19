import streamlit as st
import pandas as pd
import openai

# Title
st.title("🚚 Final Mile AI Planner")
st.caption("🔄 Last updated: 2025-05-19 | Powered by GPT-3.5")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your delivery data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📦 Delivery Data", df)

    constraints = st.text_area("⚙️ Add Route Constraints", 
        "Max 5 deliveries per route\nPrioritize 'High' priority deliveries first\nStay within time windows")

    if st.button("🧠 Generate Route Plan with GenAI"):
        # Build the LLM prompt
        prompt = f"""You are a logistics planner. Based on the delivery data and constraints below, suggest an efficient delivery route.
Also explain why the route makes sense in simple business terms.

Constraints:
{constraints}

Delivery Data (CSV):
{df.to_csv(index=False)}
"""

        # Initialize the OpenAI client with your secret key
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        try:
            # Make the API call to GPT-3.5
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )

            # Display the result
            st.subheader("🧭 AI-Suggested Route Plan")
            st.write(response.choices[0].message.content)

        except openai.RateLimitError:
            st.error("🚫 Rate limit exceeded. Try again later.")
        except openai.APIError as e:
            st.error(f"🚨 OpenAI API error: {e}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
