
import heapq
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. Streamlit Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Emergency Ambulance Route Planner",
    page_icon="🚑",
    layout="wide"
)

st.title("🚑 Emergency Ambulance Route Planner")
st.write(
    "AI-based ambulance route planning using "
    "the A* Search algorithm."
)

st.info(
    "This application uses simulated road locations "
    "and travel times for educational purposes."
)

# ------------------------------------------
# 2. Define the Road Network
# ------------------------------------------

# Each connection contains a location and travel time in minutes.

graph = {
    "Hospital": [
        ("Junction A", 4),
        ("Junction B", 2)
    ],

    "Junction A": [
        ("Hospital", 4),
        ("Junction C", 5)
    ],

    "Junction B": [
        ("Hospital", 2),
        ("Junction C", 3),
        ("Junction D", 7)
    ],

    "Junction C": [
        ("Junction A", 5),
        ("Junction B", 3),
        ("Emergency Site", 4)
    ],

    "Junction D": [
        ("Junction B", 7),
        ("Emergency Site", 2)
    ],

    "Emergency Site": [
        ("Junction C", 4),
        ("Junction D", 2)
    ]
}

# ------------------------------------------
# 3. Define the Heuristic Values
# ------------------------------------------

# Estimated remaining travel time to the destination.
# These are sample values for this demonstration.

heuristic = {
    "Hospital": 6,
    "Junction A": 5,
    "Junction B": 4,
    "Junction C": 2,
    "Junction D": 1,
    "Emergency Site": 0
}

# ------------------------------------------
# 4. A* Search Algorithm
# ------------------------------------------

def a_star(start, goal):

    # Priority queue: (f_score, g_score, current, path)
    queue = [
        (heuristic[start], 0, start, [start])
    ]

    best_cost = {}

    while queue:

        f, g, current, path = heapq.heappop(queue)

        # Destination reached
        if current == goal:
            return path, g

        # Skip a more expensive route to an explored location
        if current in best_cost and best_cost[current] <= g:
            continue

        best_cost[current] = g

        # Explore neighboring locations
        for neighbor, travel_time in graph[current]:

            new_g = g + travel_time

            if (
                neighbor not in best_cost
                or new_g < best_cost[neighbor]
            ):

                new_f = new_g + heuristic[neighbor]

                heapq.heappush(
                    queue,
                    (
                        new_f,
                        new_g,
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, float("inf")


# ------------------------------------------
# 5. Dijkstra Algorithm
# ------------------------------------------

def dijkstra(start, goal):

    queue = [(0, start, [start])]
    best_cost = {}

    while queue:

        cost, current, path = heapq.heappop(queue)

        if current == goal:
            return path, cost

        if current in best_cost and best_cost[current] <= cost:
            continue

        best_cost[current] = cost

        for neighbor, travel_time in graph[current]:

            new_cost = cost + travel_time

            if (
                neighbor not in best_cost
                or new_cost < best_cost[neighbor]
            ):

                heapq.heappush(
                    queue,
                    (
                        new_cost,
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None, float("inf")


# ------------------------------------------
# 6. User Interface
# ------------------------------------------

locations = list(graph.keys())

col1, col2 = st.columns(2)

with col1:
    start = st.selectbox(
        "Select ambulance starting location",
        locations,
        index=locations.index("Hospital")
    )

with col2:
    goal = st.selectbox(
        "Select emergency destination",
        locations,
        index=locations.index("Emergency Site")
    )

# ------------------------------------------
# 7. Find the Best Route
# ------------------------------------------

if st.button("Find Best Route", type="primary"):

    if start == goal:

        st.warning(
            "Please select different starting and "
            "destination locations."
        )

    else:

        # Run both algorithms
        route, travel_time = a_star(start, goal)

        dijkstra_route, dijkstra_time = dijkstra(
            start,
            goal
        )

        if route:

            st.success("Route calculated successfully!")

            # Display the A* result
            st.subheader("Recommended Ambulance Route")

            st.write(" ➜ ".join(route))

            # Display travel-time comparison
            col1, col2 = st.columns(2)

            col1.metric(
                "A* Estimated Travel Time",
                f"{travel_time} minutes"
            )

            col2.metric(
                "Dijkstra Estimated Travel Time",
                f"{dijkstra_time} minutes"
            )

            # ------------------------------------------
            # 8. Visualize the Road Network
            # ------------------------------------------

            st.subheader("Road Network Visualization")

            G = nx.Graph()

            for location, neighbors in graph.items():

                for neighbor, cost in neighbors:

                    G.add_edge(
                        location,
                        neighbor,
                        weight=cost
                    )

            positions = nx.spring_layout(
                G,
                seed=42
            )

            fig, ax = plt.subplots(figsize=(10, 6))

            # Draw all road connections
            nx.draw_networkx_edges(
                G,
                positions,
                edge_color="gray",
                width=1.5,
                ax=ax
            )

            # Draw locations
            nx.draw_networkx_nodes(
                G,
                positions,
                node_size=2000,
                ax=ax
            )

            nx.draw_networkx_labels(
                G,
                positions,
                font_size=9,
                ax=ax
            )

            # Highlight the recommended route
            route_edges = list(
                zip(route, route[1:])
            )

            nx.draw_networkx_edges(
                G,
                positions,
                edgelist=route_edges,
                edge_color="red",
                width=3,
                ax=ax
            )

            # Display travel times on road connections
            edge_labels = nx.get_edge_attributes(
                G,
                "weight"
            )

            nx.draw_networkx_edge_labels(
                G,
                positions,
                edge_labels=edge_labels,
                ax=ax
            )

            ax.set_title(
                "A* Search - Emergency Ambulance Route"
            )

            ax.axis("off")

            st.pyplot(fig)

            plt.close(fig)

            # ------------------------------------------
            # 9. Compare Algorithms
            # ------------------------------------------

            st.subheader("Algorithm Comparison")

            comparison = {
                "Algorithm": [
                    "A* Search",
                    "Dijkstra"
                ],
                "Route": [
                    " -> ".join(route),
                    " -> ".join(dijkstra_route)
                    if dijkstra_route else "No route"
                ],
                "Travel Time (minutes)": [
                    travel_time,
                    dijkstra_time
                ]
            }

            st.dataframe(
                comparison,
                use_container_width=True
            )

            if travel_time == dijkstra_time:

                st.info(
                    "Both algorithms found routes with "
                    "the same estimated travel cost."
                )

            else:

                st.info(
                    "The algorithms returned different "
                    "estimated travel costs. Review the "
                    "heuristic and route calculations."
                )

        else:

            st.error("No route found.")

# ------------------------------------------
# 10. Footer
# ------------------------------------------

st.divider()

st.caption(
    "M.Tech Fundamentals of Artificial Intelligence "
    "Mini-Project | A* Search"
)

st.warning(
    "Educational prototype only. Road data and travel "
    "times are simulated. Do not use for real emergency dispatch."
)
