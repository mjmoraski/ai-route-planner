import streamlit as st
import pandas as pd
import numpy as np
import openai
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Title and info
st.title("🚚 Final Mile AI Planner with OR-Tools")
st.caption("🧠 Combines optimization + GenAI explanations")

# File upload
uploaded_file = st.file_uploader("📤 Upload your delivery data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📦 Uploaded Deliveries", df)

    constraints = st.text_area("⚙️ Add Route Constraints", 
        "Max 5 deliveries per route\nPrioritize 'High' priority deliveries first\nStay within time windows")

    # Generate a mock distance matrix for demo purposes
    def create_distance_matrix(num_points):
        np.random.seed(42)
        matrix = np.random.randint(5, 50, size=(num_points, num_points))
        np.fill_diagonal(matrix, 0)
        return matrix.tolist()

    # Optimize route using OR-Tools
    def optimize_route(distance_matrix):
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            return route
        else:
            return None

    if st.button("📍 Optimize Route + Explain with AI"):
        # Build and optimize the route
        distance_matrix = create_distance_matrix(len(df))
        route = optimize_route(distance_matrix)

        if route:
            ordered_df = df.iloc[route].reset_index(drop=True)
            st.write("🧭 Optimized Route Order", ordered_df)

            # Prompt LLM for explanation
            prompt = f"""You are a logistics planner. Given the following delivery stops (already ordered for optimal routing),
explain in business-friendly terms why this route is efficient and how it benefits a driver.

Route:
{ordered_df.to_csv(index=False)}
"""

            # Call OpenAI
            client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4
                )

                st.subheader("🧠 AI Route Explanation")
                st.write(response.choices[0].message.content)

            except openai.APIError as e:
                st.error(f"OpenAI API error: {e}")
        else:
            st.error("⚠️ Unable to generate a route. Try with fewer stops.")

