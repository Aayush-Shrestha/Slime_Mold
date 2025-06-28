import numpy as np
import matplotlib.pyplot as plt
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
        self.energy = 200  # Increased energy for longer exploration
        self.alive = True
        self.moving = False
        self.move_progress = 0.0
        self.agent_id = agent_id
        self.source_id = source_id
        self.found_food = False
        self.food_memory = set()  # Remember found food nodes
        self.return_to_source_mode = False
        
    def move_towards_target(self, graph_positions, speed=0.06):
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

class SlimeMoldFoodNetwork:
    def __init__(self, num_nodes=20, num_sources=None, num_food=None, agents_per_source=25):
        self.num_nodes = num_nodes
        self.num_sources = num_sources or random.randint(2, 3)
        self.num_food = num_food or random.randint(3, 6)
        self.agents_per_source = agents_per_source
        self.graph = None
        self.positions = None
        self.agents = []
        self.source_nodes = []
        self.food_nodes = []
        self.pheromone_map = {}
        self.food_pheromone_map = {}  # Special pheromones for food paths
        self.discovered_edges = set()
        self.stable_paths = set()  # Paths that have been reinforced multiple times
        self.step_count = 0
        self.food_connections = {}  # Track connections to food
        self.setup_graph()
        
    def setup_graph(self):
        """Create a 2D graph with random weights"""
        self.graph = nx.complete_graph(self.num_nodes)
        
        # Generate 2D positions for nodes in a circular/spread pattern
        self.positions = {}
        for i in range(self.num_nodes):
            # Use a combination of circular and random positioning for better spread
            if i < self.num_nodes // 2:
                # Circular arrangement for some nodes
                angle = (2 * np.pi * i) / (self.num_nodes // 2)
                radius = random.uniform(8, 12)
                x = radius * np.cos(angle) + random.uniform(-2, 2)
                y = radius * np.sin(angle) + random.uniform(-2, 2)
            else:
                # Random positioning for others
                x = random.uniform(-15, 15)
                y = random.uniform(-15, 15)
            
            self.positions[i] = (x, y)
        
        # Assign random weights to edges
        for edge in self.graph.edges():
            weight = random.randint(1, 100)
            self.graph[edge[0]][edge[1]]['weight'] = weight
            
    def initialize_agents(self):
        """Initialize agents and designate source and food nodes"""
        # Select source nodes
        self.source_nodes = random.sample(range(self.num_nodes), self.num_sources)
        
        # Select food nodes (not overlapping with sources)
        remaining_nodes = [n for n in range(self.num_nodes) if n not in self.source_nodes]
        self.food_nodes = random.sample(remaining_nodes, self.num_food)
        
        # Initialize agents at source nodes
        self.agents = []
        agent_id = 0
        for source_id, source_node in enumerate(self.source_nodes):
            for i in range(self.agents_per_source):
                agent = SlimeAgent(source_node, self.positions, agent_id, source_id)
                # Add small random offset
                offset = np.random.normal(0, 0.1, 2)
                agent.position += offset
                self.agents.append(agent)
                agent_id += 1
                
        total_agents = len(self.agents)
        print(f"Initialized {total_agents} agents at {self.num_sources} sources")
        print(f"Source nodes: {self.source_nodes}")
        print(f"Food nodes: {self.food_nodes}")
        return self.source_nodes, self.food_nodes
        
    def get_edge_attractiveness(self, agent, from_node, to_node):
        """Calculate edge attractiveness based on multiple factors"""
        edge = tuple(sorted([from_node, to_node]))
        weight = self.graph[from_node][to_node]['weight']
        
        # Regular pheromone
        pheromone = self.pheromone_map.get(edge, 0.1)
        
        # Food-specific pheromone (stronger signal)
        food_pheromone = self.food_pheromone_map.get(edge, 0)
        
        # Base attractiveness
        attractiveness = ((pheromone + food_pheromone * 2) ** 1.3) / (weight ** 0.8)
        
        # Food attraction bonus
        if to_node in self.food_nodes:
            if not agent.found_food or to_node not in agent.food_memory:
                attractiveness *= 3.0  # Strong attraction to new food
            else:
                attractiveness *= 1.5  # Moderate attraction to known food
                
        # Source attraction when returning
        if agent.return_to_source_mode and to_node in self.source_nodes:
            attractiveness *= 2.0
            
        # Stable path bonus
        if edge in self.stable_paths:
            attractiveness *= 1.8
            
        return attractiveness
        
    def select_next_node(self, agent):
        """Select next node with food-seeking and exploration balance"""
        current = agent.current_node
        neighbors = list(self.graph.neighbors(current))
        
        if not neighbors:
            return None
        
        # Check if agent found food
        if current in self.food_nodes and not agent.found_food:
            agent.found_food = True
            agent.food_memory.add(current)
            agent.return_to_source_mode = True
            print(f"Agent {agent.agent_id} found food at node {current}!")
            
        # If returning to source and reached it, reset mode
        if agent.return_to_source_mode and current in self.source_nodes:
            agent.return_to_source_mode = False
            agent.energy += 50  # Reward for completing food-source cycle
            
        # Balance exploration vs exploitation
        exploration_factor = 0.3 if agent.found_food else 0.7
        
        # Remove recent path to encourage exploration
        recent_path = agent.path[-8:] if len(agent.path) > 8 else agent.path
        
        # Separate neighbors into explored and unexplored
        unexplored = [n for n in neighbors if n not in recent_path[:-1]]
        explored = [n for n in neighbors if n in recent_path[:-1]]
        
        # Choose neighbor set based on exploration factor
        if unexplored and random.random() < exploration_factor:
            neighbors = unexplored
        else:
            neighbors = neighbors  # Use all neighbors
            
        # Calculate attractiveness for each neighbor
        attractiveness = []
        for neighbor in neighbors:
            attr = self.get_edge_attractiveness(agent, current, neighbor)
            
            # Add exploration bonus for less visited nodes
            visit_count = agent.path.count(neighbor)
            exploration_bonus = 1.0 / (1 + visit_count * 0.3)
            attr *= exploration_bonus
            
            # Add randomness for exploration
            attr *= random.uniform(0.7, 1.3)
            attractiveness.append(attr)
            
        # Normalize and select
        total = sum(attractiveness)
        if total == 0:
            return random.choice(neighbors)
            
        probabilities = [a / total for a in attractiveness]
        
        # Probabilistic selection
        r = random.random()
        cumulative = 0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return neighbors[i]
                
        return neighbors[-1]
        
    def update_pheromones(self, agent, from_node, to_node):
        """Update pheromone levels with food-specific bonuses"""
        edge = tuple(sorted([from_node, to_node]))
        weight = self.graph[from_node][to_node]['weight']
        
        # Base pheromone deposit
        deposit = max(1.0, 150 / weight)
        
        # Bonus for food-related paths
        food_bonus = 1.0
        if from_node in self.food_nodes or to_node in self.food_nodes:
            food_bonus = 2.5
            # Add to food pheromone map
            self.food_pheromone_map[edge] = self.food_pheromone_map.get(edge, 0) + deposit * 1.5
            
        # Bonus for agents that have found food
        if agent.found_food:
            food_bonus *= 1.5
            
        final_deposit = deposit * food_bonus
        
        # Update regular pheromone map
        if edge in self.pheromone_map:
            self.pheromone_map[edge] += final_deposit
        else:
            self.pheromone_map[edge] = final_deposit
            
        # Track stable paths (heavily used edges)
        if self.pheromone_map[edge] > 100:  # Threshold for stable path
            self.stable_paths.add(edge)
            
        # Add to discovered edges
        self.discovered_edges.add(edge)
        
        # Track food connections
        if from_node in self.food_nodes or to_node in self.food_nodes:
            food_node = from_node if from_node in self.food_nodes else to_node
            other_node = to_node if from_node in self.food_nodes else from_node
            
            if food_node not in self.food_connections:
                self.food_connections[food_node] = set()
            self.food_connections[food_node].add(other_node)
        
    def evaporate_pheromones(self, rate=0.995):
        """Evaporate pheromones with slower decay for food paths"""
        # Regular pheromones
        for edge in list(self.pheromone_map.keys()):
            self.pheromone_map[edge] *= rate
            if self.pheromone_map[edge] < 0.01:
                del self.pheromone_map[edge]
                
        # Food pheromones (slower decay)
        food_rate = 0.998
        for edge in list(self.food_pheromone_map.keys()):
            self.food_pheromone_map[edge] *= food_rate
            if self.food_pheromone_map[edge] < 0.01:
                del self.food_pheromone_map[edge]
                
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
                    continue
                    
            # Agent reached target, select new target
            if not agent.moving:
                next_node = self.select_next_node(agent)
                
                if next_node is not None and next_node != agent.current_node:
                    # Update pheromones
                    self.update_pheromones(agent, agent.current_node, next_node)
                    
                    # Set new target
                    agent.set_target(next_node)
                    agent.energy -= 1
                    
                    if agent.energy <= 0:
                        agent.alive = False
                        
        # Evaporate pheromones less frequently
        if self.step_count % 8 == 0:
            self.evaporate_pheromones()
            
        self.step_count += 1
        return len([a for a in self.agents if a.alive and a.energy > 0]) > 0
        
    def animate_simulation(self, max_steps=400):
        """Run simulation with real-time 2D animation"""
        print("Starting 2D slime mold food network simulation...")
        source_nodes, food_nodes = self.initialize_agents()
        
        # Set up the figure and 2D axis
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        
        def animate(frame):
            if frame < max_steps and self.step_simulation():
                self.update_plot(source_nodes, food_nodes, frame)
            else:
                # Simulation ended
                self.update_plot(source_nodes, food_nodes, frame, simulation_ended=True)
                
            return []
            
        # Create animation
        self.anim = FuncAnimation(
            self.fig, animate, frames=max_steps + 50, 
            interval=80, blit=False, repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
    def update_plot(self, source_nodes, food_nodes, frame, simulation_ended=False):
        """Update the 2D plot for animation"""
        self.ax.clear()
        
        # Plot discovered edges with pheromone intensity
        if self.discovered_edges and self.pheromone_map:
            max_pheromone = max(self.pheromone_map.values()) if self.pheromone_map else 1
            max_food_pheromone = max(self.food_pheromone_map.values()) if self.food_pheromone_map else 1
            
            for edge in self.discovered_edges:
                node1, node2 = edge
                pos1 = self.positions[node1]
                pos2 = self.positions[node2]
                
                # Regular pheromone visualization
                if edge in self.pheromone_map:
                    pheromone = self.pheromone_map[edge]
                    intensity = min(1.0, pheromone / max_pheromone)
                    
                    # Different colors for different path types
                    if edge in self.stable_paths:
                        color = 'darkgreen'
                        linewidth = 1.5 + intensity * 3
                        alpha = 0.6 + intensity * 0.4
                    elif edge in self.food_pheromone_map:
                        color = 'orange'
                        linewidth = 1.0 + intensity * 2.5
                        alpha = 0.4 + intensity * 0.5
                    else:
                        color = 'purple'
                        linewidth = 0.5 + intensity * 2
                        alpha = 0.2 + intensity * 0.4
                        
                    self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                               c=color, linewidth=linewidth, alpha=alpha)
        
        # Plot all nodes with different colors for sources and food
        for i, (node, pos) in enumerate(self.positions.items()):
            if node in source_nodes:
                self.ax.scatter(*pos, c='red', s=300, alpha=0.9, marker='s', 
                              edgecolors='darkred', linewidth=2, label='Source' if i == source_nodes[0] else '')
            elif node in food_nodes:
                self.ax.scatter(*pos, c='gold', s=250, alpha=0.9, marker='^', 
                              edgecolors='orange', linewidth=2, label='Food' if i == food_nodes[0] else '')
            else:
                self.ax.scatter(*pos, c='lightblue', s=100, alpha=0.7, 
                              edgecolors='blue', linewidth=1)
                
            # Node labels
            self.ax.text(pos[0], pos[1] + 0.8, str(node), fontsize=9, 
                        fontweight='bold', ha='center')
        
        # Plot agents with different colors per source
        agent_colors = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta']
        active_agents = [a for a in self.agents if a.alive and a.energy > 0]
        
        for agent in active_agents:
            color = agent_colors[agent.source_id % len(agent_colors)]
            marker = 'o' if not agent.found_food else '*'
            size = 80 if not agent.found_food else 120
            alpha = 0.8 if not agent.return_to_source_mode else 1.0
            
            self.ax.scatter(*agent.position, c=color, s=size, alpha=alpha, 
                          marker=marker, edgecolors='black', linewidth=1)
        
        # Set plot properties
        self.ax.set_xlabel('X Position', fontsize=12)
        self.ax.set_ylabel('Y Position', fontsize=12)
        
        # Title with comprehensive information
        active_count = len(active_agents)
        food_found = sum(1 for a in self.agents if a.found_food)
        stable_count = len(self.stable_paths)
        
        title = f"Slime Mold Food Network - Step {frame}\n"
        if simulation_ended:
            title += f"SIMULATION COMPLETE - "
        title += f"Active Agents: {active_count}, Food Found: {food_found}, "
        title += f"Edges: {len(self.discovered_edges)}, Stable Paths: {stable_count}"
        
        self.ax.set_title(title, fontsize=11, pad=20)
        
        # Legend
        if frame == 0:
            self.ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Set axis limits with margin
        all_positions = np.array(list(self.positions.values()))
        margin = 3
        self.ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
        self.ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
        
        # Grid for better visualization
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

def main():
    """Main function to run the animated simulation"""
    # Create and run 2D food network simulation
    sim = SlimeMoldFoodNetwork(
        num_nodes=20, 
        num_sources=None,  # Random 2-3 sources
        num_food=None,     # Random 3-6 food nodes
        agents_per_source=30
    )
    sim.animate_simulation(max_steps=350)
    
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Total nodes: {sim.num_nodes}")
    print(f"Source nodes: {sim.num_sources} - {sim.source_nodes}")
    print(f"Food nodes: {sim.num_food} - {sim.food_nodes}")
    print(f"Edges discovered: {len(sim.discovered_edges)}")
    print(f"Stable paths formed: {len(sim.stable_paths)}")
    print(f"Food connections established: {len(sim.food_connections)}")
    
    # Analyze food network
    agents_found_food = sum(1 for a in sim.agents if a.found_food)
    print(f"Agents that found food: {agents_found_food}/{len(sim.agents)}")
    
    if sim.food_connections:
        print("\nFood Network Connections:")
        for food_node, connections in sim.food_connections.items():
            print(f"  Food {food_node}: connected to {len(connections)} nodes")

if __name__ == "__main__":
    main()