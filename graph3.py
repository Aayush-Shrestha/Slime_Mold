import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import random
from matplotlib.animation import FuncAnimation
import time

class SlimeAgent:
    def __init__(self, start_node, graph_positions, agent_id=0, source_id=0):
        self.position = np.array(graph_positions[start_node])
        self.current_node = start_node
        self.target_node = start_node
        self.path = [start_node]
        self.visited_edges = set()
        self.pheromone_trail = {}
        self.energy = 150  # Reduced energy since we have more agents
        self.alive = True
        self.moving = False
        self.move_progress = 0.0
        self.agent_id = agent_id
        self.source_id = source_id  # Which source this agent came from
        
    def move_towards_target(self, graph_positions, speed=0.08):
        """Smoothly move agent towards target node"""
        if not self.moving or self.target_node == self.current_node:
            return True
            
        start_pos = np.array(graph_positions[self.current_node])
        target_pos = np.array(graph_positions[self.target_node])
        
        self.move_progress += speed
        
        if self.move_progress >= 1.0:
            # Reached target
            self.position = target_pos
            self.current_node = self.target_node
            self.path.append(self.target_node)
            self.moving = False
            self.move_progress = 0.0
            return True
        else:
            # Interpolate position
            self.position = start_pos + (target_pos - start_pos) * self.move_progress
            return False
            
    def set_target(self, target_node):
        """Set new target node for movement"""
        if target_node != self.current_node:
            self.target_node = target_node
            self.moving = True
            self.move_progress = 0.0

class SlimeMoldMST:
    def __init__(self, num_nodes=26, num_sources=None, agents_per_source=40):
        self.num_nodes = num_nodes
        self.num_sources = num_sources or random.randint(2, 4)
        self.agents_per_source = agents_per_source
        self.graph = None
        self.positions = None
        self.agents = []
        self.source_nodes = []
        self.pheromone_map = {}
        self.discovered_edges = set()
        self.mst_edges = []
        self.step_count = 0
        self.animation_data = {'positions': [], 'edges': [], 'pheromones': []}
        self.setup_graph()
        
    def setup_graph(self):
        """Create a fully connected 3D graph with random weights"""
        self.graph = nx.complete_graph(self.num_nodes)
        
        # Generate 3D positions for nodes in a more spread out manner
        self.positions = {}
        for i in range(self.num_nodes):
            # Use spherical distribution for better 3D visualization
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            r = random.uniform(5, 15)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            self.positions[i] = (x, y, z)
        
        # Assign random weights to edges
        for edge in self.graph.edges():
            weight = random.randint(1, 100)
            self.graph[edge[0]][edge[1]]['weight'] = weight
            
    def initialize_agents(self):
        """Initialize multiple slime agents at each random source node"""
        self.source_nodes = random.sample(range(self.num_nodes), self.num_sources)
        self.agents = []
        
        agent_id = 0
        for source_id, source_node in enumerate(self.source_nodes):
            for i in range(self.agents_per_source):
                agent = SlimeAgent(source_node, self.positions, agent_id, source_id)
                # Add small random offset to prevent all agents from being at exact same position
                offset = np.random.normal(0, 0.1, 3)
                agent.position += offset
                self.agents.append(agent)
                agent_id += 1
                
        total_agents = len(self.agents)
        print(f"Initialized {total_agents} agents ({self.agents_per_source} per source)")
        print(f"Source nodes: {self.source_nodes}")
        return self.source_nodes
        
    def get_edge_attractiveness(self, from_node, to_node):
        """Calculate edge attractiveness based on weight and pheromones"""
        edge = tuple(sorted([from_node, to_node]))
        weight = self.graph[from_node][to_node]['weight']
        pheromone = self.pheromone_map.get(edge, 0.1)
        
        # Lower weight and higher pheromone = more attractive
        attractiveness = (pheromone ** 1.2) / (weight ** 0.9)
        return attractiveness
        
    def select_next_node(self, agent):
        """Select next node for agent based on attractiveness"""
        current = agent.current_node
        neighbors = list(self.graph.neighbors(current))
        
        if not neighbors:
            return None
            
        # Remove recently visited nodes to encourage exploration
        recent_path = agent.path[-5:] if len(agent.path) > 5 else agent.path
        unvisited_neighbors = [n for n in neighbors if n not in recent_path[:-1]]
        
        if unvisited_neighbors:
            neighbors = unvisited_neighbors
            
        # Calculate probabilities for each neighbor
        attractiveness = []
        for neighbor in neighbors:
            attr = self.get_edge_attractiveness(current, neighbor)
            # Add exploration bonus for less visited nodes
            visit_count = agent.path.count(neighbor)
            exploration_bonus = 1.0 / (1 + visit_count * 0.5)
            attr *= exploration_bonus
            # Add randomness
            attr *= random.uniform(0.8, 1.2)
            attractiveness.append(attr)
            
        # Normalize probabilities
        total = sum(attractiveness)
        if total == 0:
            return random.choice(neighbors)
            
        probabilities = [a / total for a in attractiveness]
        
        # Select based on probability
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return neighbors[i]
                
        return neighbors[-1]
        
    def update_pheromones(self, agent, from_node, to_node):
        """Update pheromone levels on traversed edge"""
        edge = tuple(sorted([from_node, to_node]))
        weight = self.graph[from_node][to_node]['weight']
        
        # Higher pheromone deposit for lower weight edges
        deposit = max(1.0, 200 / weight)
        
        if edge in self.pheromone_map:
            self.pheromone_map[edge] += deposit
        else:
            self.pheromone_map[edge] = deposit
            
        # Add to discovered edges
        self.discovered_edges.add(edge)
        
    def evaporate_pheromones(self, rate=0.98):
        """Evaporate pheromones over time"""
        for edge in list(self.pheromone_map.keys()):
            self.pheromone_map[edge] *= rate
            if self.pheromone_map[edge] < 0.01:
                del self.pheromone_map[edge]
                
    def step_simulation(self):
        """Single step of the simulation"""
        active_agents = [a for a in self.agents if a.alive and a.energy > 0]
        
        if not active_agents:
            return False
            
        for agent in active_agents:
            # Handle agent movement
            if agent.moving:
                reached = agent.move_towards_target(self.positions)
                if not reached:
                    continue  # Still moving, skip other logic
                    
            # Agent has reached current target, select new target
            if not agent.moving:
                next_node = self.select_next_node(agent)
                
                if next_node is not None and next_node != agent.current_node:
                    # Update pheromones for the edge we're about to traverse
                    self.update_pheromones(agent, agent.current_node, next_node)
                    
                    # Set new target
                    agent.set_target(next_node)
                    agent.energy -= 1
                    
                    if agent.energy <= 0:
                        agent.alive = False
                        
        # Evaporate pheromones
        if self.step_count % 5 == 0:  # Less frequent evaporation
            self.evaporate_pheromones()
            
        self.step_count += 1
        return len([a for a in self.agents if a.alive and a.energy > 0]) > 0
        
    def extract_mst(self):
        """Extract MST from discovered edges with pheromone weights"""
        if len(self.discovered_edges) < self.num_nodes - 1:
            print("Not enough edges discovered for complete MST")
            return
            
        # Create subgraph with discovered edges
        mst_graph = nx.Graph()
        
        for edge in self.discovered_edges:
            node1, node2 = edge
            pheromone = self.pheromone_map.get(edge, 0.01)
            # Use inverse of pheromone as weight (higher pheromone = lower weight)
            mst_weight = 1.0 / (pheromone + 0.01)
            mst_graph.add_edge(node1, node2, weight=mst_weight)
            
        # Find MST using Kruskal's algorithm
        try:
            # Ensure graph is connected
            if nx.is_connected(mst_graph):
                mst = nx.minimum_spanning_tree(mst_graph)
                self.mst_edges = list(mst.edges())
                print(f"MST found with {len(self.mst_edges)} edges")
            else:
                # Find MST of largest connected component
                largest_cc = max(nx.connected_components(mst_graph), key=len)
                subgraph = mst_graph.subgraph(largest_cc)
                mst = nx.minimum_spanning_tree(subgraph)
                self.mst_edges = list(mst.edges())
                print(f"MST found for largest component with {len(self.mst_edges)} edges")
        except Exception as e:
            print(f"Could not form MST: {e}")
            
    def animate_simulation(self, max_steps=300):
        """Run simulation with real-time animation"""
        print("Starting animated slime mold MST simulation...")
        source_nodes = self.initialize_agents()
        
        # Set up the figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        self.node_scatter = None
        self.agent_scatter = None
        self.edge_lines = []
        self.mst_lines = []
        self.pheromone_lines = []
        
        def animate(frame):
            if frame < max_steps and self.step_simulation():
                self.update_plot(source_nodes, frame)
            elif frame == max_steps or not any(a.alive and a.energy > 0 for a in self.agents):
                # Simulation ended, extract and show MST
                if not self.mst_edges:
                    self.extract_mst()
                self.update_plot(source_nodes, frame, show_mst=True)
                
            return []
            
        # Create animation
        self.anim = FuncAnimation(
            self.fig, animate, frames=max_steps + 50, 
            interval=100, blit=False, repeat=False
        )
        
        plt.show()
        
    def update_plot(self, source_nodes, frame, show_mst=False):
        """Update the 3D plot for animation"""
        self.ax.clear()
        
        # Plot all nodes
        node_colors = ['red' if i in source_nodes else 'lightblue' for i in range(self.num_nodes)]
        node_sizes = [150 if i in source_nodes else 100 for i in range(self.num_nodes)]
        
        for i, (node, pos) in enumerate(self.positions.items()):
            self.ax.scatter(*pos, c=node_colors[i], s=node_sizes[i], alpha=0.8)
            self.ax.text(pos[0], pos[1], pos[2], str(node), fontsize=8, fontweight='bold')
            
        # Plot discovered edges with pheromone intensity
        if self.discovered_edges and self.pheromone_map:
            max_pheromone = max(self.pheromone_map.values()) if self.pheromone_map else 1
            
            for edge in self.discovered_edges:
                if edge in self.pheromone_map:
                    node1, node2 = edge
                    pos1 = self.positions[node1]
                    pos2 = self.positions[node2]
                    
                    pheromone = self.pheromone_map[edge]
                    intensity = min(1.0, pheromone / max_pheromone)
                    
                    self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                               c='purple', linewidth=0.5 + intensity * 2, alpha=0.3 + intensity * 0.5)
        
        # Plot agents
        agent_positions = []
        agent_colors = []
        colors = ['red', 'green', 'blue', 'orange', 'yellow', 'cyan']
        
        for i, agent in enumerate(self.agents):
            if agent.alive and agent.energy > 0:
                agent_positions.append(agent.position)
                agent_colors.append(colors[i % len(colors)])
                
        if agent_positions:
            agent_positions = np.array(agent_positions)
            for i, pos in enumerate(agent_positions):
                self.ax.scatter(*pos, c=agent_colors[i], s=200, alpha=0.9, 
                              marker='o', edgecolors='black', linewidth=2)
                
        # Show MST overlay if requested
        if show_mst and self.mst_edges:
            for edge in self.mst_edges:
                node1, node2 = edge
                pos1 = self.positions[node1]
                pos2 = self.positions[node2]
                
                self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                           c='red', linewidth=4, alpha=0.9)
                           
        # Set plot properties
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        title = f"Slime Mold MST Simulation - Step {frame}"
        if show_mst:
            title += f" - MST Complete ({len(self.mst_edges)} edges)"
        else:
            active_agents = sum(1 for a in self.agents if a.alive and a.energy > 0)
            title += f" - Active Agents: {active_agents}, Edges: {len(self.discovered_edges)}"
            
        self.ax.set_title(title)
        
        # Set consistent axis limits
        all_positions = np.array(list(self.positions.values()))
        margin = 2
        self.ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
        self.ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
        self.ax.set_zlim(all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin)

def main():
    """Main function to run the animated simulation"""
    # Create and run animated simulation
    sim = SlimeMoldMST(num_nodes=26, num_sources=None)  # Random number of sources (2-4)
    sim.animate_simulation(max_steps=250)
    
    print("\nSimulation Summary:")
    print(f"- Total nodes: {sim.num_nodes}")
    print(f"- Source nodes: {sim.num_sources}")
    print(f"- Edges discovered: {len(sim.discovered_edges)}")
    print(f"- MST edges: {len(sim.mst_edges)}")
    
    if sim.mst_edges:
        total_weight = sum(sim.graph[edge[0]][edge[1]]['weight'] for edge in sim.mst_edges)
        print(f"- MST total weight: {total_weight}")

if __name__ == "__main__":
    main()