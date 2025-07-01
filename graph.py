import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button # Import Button
import time

class SlimeAgent:
    def __init__(self, start_node, graph_positions, agent_id=0, source_id=0):
        self.position = np.array(graph_positions[start_node][:2])  # 2D only
        self.current_node = start_node
        self.target_node = start_node
        self.path = [start_node]
        self.visited_edges = set()
        self.pheromone_trail = {}
        self.energy = 200
        self.alive = True
        self.moving = False
        self.move_progress = 0.0
        self.agent_id = agent_id
        self.source_id = source_id
        self.food_seeking = True  # Whether agent is seeking food
        self.has_found_food = False
        self.target_food = None
        self.exploration_timer = 0
        
    def move_towards_target(self, graph_positions, speed=0.06):
        """Smoothly move agent towards target node"""
        if not self.moving or self.target_node == self.current_node:
            return True
            
        start_pos = np.array(graph_positions[self.current_node][:2])
        target_pos = np.array(graph_positions[self.target_node][:2])
        
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
    def __init__(self, num_nodes=26, num_sources=None, num_food=None, agents_per_source=40):
        self.num_nodes = num_nodes
        self.num_sources = num_sources or random.randint(3, 4)
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
        self.stable_network_edges = set()  # Edges that form stable paths
        self.step_count = 0
        self.setup_graph()
        
        # For animation control
        self.current_frame = 0
        self.paused = False
        self.history = [] # To store states for 'prev' button
        self.max_history_length = 500 # Limit history to prevent memory issues

    def setup_graph(self):
        """Create a fully connected 2D graph with random weights"""
        self.graph = nx.complete_graph(self.num_nodes)
        
        # Generate 2D positions for nodes in a circular/scattered pattern
        self.positions = {}
        for i in range(self.num_nodes):
            # Use a mix of circular and random distribution
            if i < self.num_nodes // 2:
                # Circular arrangement
                angle = 2 * np.pi * i / (self.num_nodes // 2)
                radius = random.uniform(8, 12)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            else:
                # Random scattered
                x = random.uniform(-15, 15)
                y = random.uniform(-15, 15)
            
            self.positions[i] = (x, y)
        
        # Assign random weights to edges (distance-based with randomness)
        for edge in self.graph.edges():
            node1, node2 = edge
            pos1 = np.array(self.positions[node1])
            pos2 = np.array(self.positions[node2])
            distance = np.linalg.norm(pos1 - pos2)
            # Weight based on distance with some randomness
            weight = int(distance * random.uniform(0.8, 1.5)) + random.randint(1, 20)
            weight = max(1, min(100, weight))  # Clamp between 1-100
            self.graph[edge[0]][edge[1]]['weight'] = weight
            
    def initialize_nodes(self):
        """Initialize source and food nodes"""
        all_nodes = list(range(self.num_nodes))
        
        # Select source nodes
        self.source_nodes = random.sample(all_nodes, self.num_sources)
        remaining_nodes = [n for n in all_nodes if n not in self.source_nodes]
        
        # Select food nodes from remaining
        self.food_nodes = random.sample(remaining_nodes, self.num_food)
        
        print(f"Source nodes: {self.source_nodes}")
        print(f"Food nodes: {self.food_nodes}")
        
    def initialize_agents(self):
        """Initialize multiple slime agents at each source node"""
        self.agents = []
        
        agent_id = 0
        for source_id, source_node in enumerate(self.source_nodes):
            for i in range(self.agents_per_source):
                agent = SlimeAgent(source_node, self.positions, agent_id, source_id)
                # Add small random offset to prevent all agents from being at exact same position
                offset = np.random.normal(0, 0.2, 2)
                agent.position += offset
                # Assign random target food for diversity
                agent.target_food = random.choice(self.food_nodes)
                self.agents.append(agent)
                agent_id += 1
                
        total_agents = len(self.agents)
        print(f"Initialized {total_agents} agents ({self.agents_per_source} per source)")
        
    def get_food_attractiveness(self, from_node, to_node, agent):
        """Calculate attractiveness towards food nodes"""
        edge = tuple(sorted([from_node, to_node]))
        weight = self.graph[from_node][to_node]['weight']
        
        # Regular pheromone
        pheromone = self.pheromone_map.get(edge, 0.1)
        # Food-specific pheromone (stronger)
        food_pheromone = self.food_pheromone_map.get(edge, 0.1)
        
        # Distance to target food
        food_pos = np.array(self.positions[agent.target_food][:2])
        to_pos = np.array(self.positions[to_node][:2])
        from_pos = np.array(self.positions[from_node][:2])
        
        # Distance improvement towards food
        current_dist_to_food = np.linalg.norm(from_pos - food_pos)
        new_dist_to_food = np.linalg.norm(to_pos - food_pos)
        distance_improvement = max(0, current_dist_to_food - new_dist_to_food)
        
        # Food attraction factor
        food_attraction = 1.0
        if to_node in self.food_nodes:
            food_attraction = 5.0  # Strong attraction to any food
        if to_node == agent.target_food:
            food_attraction = 10.0  # Very strong attraction to target food
            
        # Combined attractiveness
        attractiveness = (
            (pheromone + food_pheromone * 2) ** 1.3 * food_attraction * (1 + distance_improvement * 0.5)
        ) / (weight ** 0.7)
        
        return attractiveness
        
    def select_next_node(self, agent):
        """Select next node for agent based on food-seeking behavior"""
        current = agent.current_node
        neighbors = list(self.graph.neighbors(current))
        
        if not neighbors:
            return None
            
        # If agent has found food, encourage return journey with some exploration
        if current in self.food_nodes:
            agent.has_found_food = True
            agent.food_seeking = False
            
        # Mix of food-seeking and exploration behavior
        if agent.food_seeking or random.random() < 0.7:  # 70% food-seeking behavior
            # Food-seeking behavior
            attractiveness = []
            for neighbor in neighbors:
                attr = self.get_food_attractiveness(current, neighbor, agent)
                # Add exploration bonus for less visited nodes
                visit_count = agent.path.count(neighbor)
                exploration_bonus = 1.0 / (1 + visit_count * 0.3)
                attr *= exploration_bonus
                # Add randomness
                attr *= random.uniform(0.9, 1.1)
                attractiveness.append(attr)
        else:
            # Pure exploration behavior
            recent_path = agent.path[-8:] if len(agent.path) > 8 else agent.path
            unvisited_neighbors = [n for n in neighbors if n not in recent_path]
            
            if unvisited_neighbors:
                return random.choice(unvisited_neighbors)
            else:
                return random.choice(neighbors)
            
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
        
        # Regular pheromone deposit
        deposit = max(1.0, 150 / weight)
        
        if edge in self.pheromone_map:
            self.pheromone_map[edge] += deposit
        else:
            self.pheromone_map[edge] = deposit
            
        # Special food pheromone if connected to food or agent found food
        food_deposit = 0
        if from_node in self.food_nodes or to_node in self.food_nodes:
            food_deposit = deposit * 3  # Triple strength for food connections
        elif agent.has_found_food:
            food_deposit = deposit * 2  # Double strength for return paths
            
        if food_deposit > 0:
            if edge in self.food_pheromone_map:
                self.food_pheromone_map[edge] += food_deposit
            else:
                self.food_pheromone_map[edge] = food_deposit
            
        # Add to discovered edges
        self.discovered_edges.add(edge)
        
        # Track stable network edges (high pheromone concentration)
        total_pheromone = self.pheromone_map.get(edge, 0) + self.food_pheromone_map.get(edge, 0)
        if total_pheromone > 500:  # Threshold for stable path
            self.stable_network_edges.add(edge)
        
    def evaporate_pheromones(self, rate=0.985):
        """Evaporate pheromones over time"""
        for edge in list(self.pheromone_map.keys()):
            self.pheromone_map[edge] *= rate
            if self.pheromone_map[edge] < 0.1:
                del self.pheromone_map[edge]
                
        for edge in list(self.food_pheromone_map.keys()):
            self.food_pheromone_map[edge] *= rate * 0.99  # Slower evaporation for food pheromones
            if self.food_pheromone_map[edge] < 0.1:
                del self.food_pheromone_map[edge]
                
    def _get_current_state(self):
        """Captures the current state of the simulation for history."""
        # We need to capture enough info to reconstruct the display
        # and potentially rollback a step.
        # This can be memory intensive, so consider what's truly needed.
        return {
            'step_count': self.step_count,
            'pheromone_map': dict(self.pheromone_map),
            'food_pheromone_map': dict(self.food_pheromone_map),
            'discovered_edges': set(self.discovered_edges),
            'stable_network_edges': set(self.stable_network_edges),
            'agents_state': [
                {
                    'position': agent.position.copy(),
                    'current_node': agent.current_node,
                    'target_node': agent.target_node,
                    'path': list(agent.path), # Store path for exploration bonus etc.
                    'energy': agent.energy,
                    'alive': agent.alive,
                    'moving': agent.moving,
                    'move_progress': agent.move_progress,
                    'food_seeking': agent.food_seeking,
                    'has_found_food': agent.has_found_food,
                    'target_food': agent.target_food
                }
                for agent in self.agents
            ]
        }

    def _load_state(self, state):
        """Loads a previously saved state."""
        self.step_count = state['step_count']
        self.pheromone_map = state['pheromone_map']
        self.food_pheromone_map = state['food_pheromone_map']
        self.discovered_edges = state['discovered_edges']
        self.stable_network_edges = state['stable_network_edges']
        
        for i, agent_state in enumerate(state['agents_state']):
            agent = self.agents[i] # Assuming agents list order doesn't change
            agent.position = agent_state['position']
            agent.current_node = agent_state['current_node']
            agent.target_node = agent_state['target_node']
            agent.path = agent_state['path']
            agent.energy = agent_state['energy']
            agent.alive = agent_state['alive']
            agent.moving = agent_state['moving']
            agent.move_progress = agent_state['move_progress']
            agent.food_seeking = agent_state['food_seeking']
            agent.has_found_food = agent_state['has_found_food']
            agent.target_food = agent_state['target_food']

    def step_simulation(self):
        """Single step of the simulation"""
        if self.current_frame >= self.max_steps: # Stop if max steps reached
            return False

        # Store current state BEFORE stepping for 'prev' button
        self.history.append(self._get_current_state())
        if len(self.history) > self.max_history_length:
            self.history.pop(0) # Remove oldest state

        active_agents = [a for a in self.agents if a.alive and a.energy > 0]
        
        if not active_agents and self.step_count > 0: # Simulation finished
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
                    
                    # Refresh energy if at food source
                    if agent.current_node in self.food_nodes:
                        agent.energy = min(200, agent.energy + 50)
                        
                    if agent.energy <= 0:
                        agent.alive = False
                        
        # Evaporate pheromones less frequently
        if self.step_count % 8 == 0:
            self.evaporate_pheromones()
            
        self.step_count += 1
        self.current_frame += 1 # Update internal frame counter
        return len([a for a in self.agents if a.alive and a.energy > 0]) > 0
        
    def animate_simulation(self, max_steps=400):
        """Run simulation with real-time animation"""
        print("Starting animated slime mold food network simulation...")
        self.max_steps = max_steps # Store max_steps
        self.initialize_nodes()
        self.initialize_agents()
        
        # Set up the figure and 2D axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2) # Make space for buttons

        # Button axes
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_play_pause = plt.axes([0.58, 0.05, 0.1, 0.075]) # Play/Pause button

        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_play_pause = Button(ax_play_pause, 'Pause')

        def _prev(event):
            if self.current_frame > 0 and len(self.history) > 1:
                self.paused = True # Pause when using prev/next
                self.anim.event_source.stop() # Stop animation if playing
                
                # Load the state BEFORE the current one
                # Need to be careful here if history doesn't exactly match step_count
                if self.current_frame > 1 and len(self.history) >= (self.current_frame - 1):
                     # If history is full and popped old states, might need to reload from current history[-2]
                    if self.history[-1]['step_count'] == self.step_count: # If current state is in history
                         self._load_state(self.history.pop(-2)) # Load the state before current, remove current
                    else: # current state was not saved, load from history end
                        self._load_state(self.history.pop(-1)) # Load previous state, remove it

                self.current_frame = self.step_count # Update frame number to match loaded state
                self.update_plot(self.current_frame)
                self.btn_play_pause.label.set_text('Play')


        def _next(event):
            if not self.paused: # If playing, next button click should pause
                self.paused = True
                self.anim.event_source.stop()
                self.btn_play_pause.label.set_text('Play')

            # Advance simulation one step
            if self.step_simulation(): # Step and update
                self.update_plot(self.current_frame)
            else:
                # Handle end of simulation
                self.update_plot(self.current_frame, final=True)


        def _play_pause(event):
            if self.paused:
                self.paused = False
                self.anim.event_source.start()
                self.btn_play_pause.label.set_text('Pause')
            else:
                self.paused = True
                self.anim.event_source.stop()
                self.btn_play_pause.label.set_text('Play')

        self.btn_prev.on_clicked(_prev)
        self.btn_next.on_clicked(_next)
        self.btn_play_pause.on_clicked(_play_pause)
        
        def animate(frame):
            if self.paused:
                return [] # Do nothing if paused

            # Only advance simulation if current_frame matches step_count
            # This prevents stepping twice if _next was clicked
            if frame >= self.current_frame:
                if self.step_simulation():
                    pass # Simulation advanced successfully
                else:
                    # Simulation ended
                    self.anim.event_source.stop() # Stop animation
                    self.update_plot(self.current_frame, final=True)
                    return [] # No need to re-draw if already stopped

            self.update_plot(self.current_frame)
            return [] # No blitting required

        # Initial plot to show the starting state
        self.update_plot(0)

        # Create animation
        self.anim = FuncAnimation(
            self.fig, animate, frames=max_steps + 50, # Provide enough frames, actual control is internal
            interval=10, blit=False, repeat=False
        )
        
        plt.show()
        
    def update_plot(self, frame, final=False):
        """Update the 2D plot for animation"""
        self.ax.clear()
        
        # Plot all discovered edges with pheromone intensity
        if self.discovered_edges:
            all_pheromones = {}
            for edge in self.discovered_edges:
                regular = self.pheromone_map.get(edge, 0)
                food = self.food_pheromone_map.get(edge, 0)
                all_pheromones[edge] = regular + food
                
            if all_pheromones:
                max_pheromone = max(all_pheromones.values())
                
                for edge in self.discovered_edges:
                    if edge in all_pheromones:
                        node1, node2 = edge
                        pos1 = self.positions[node1]
                        pos2 = self.positions[node2]
                        
                        total_pheromone = all_pheromones[edge]
                        intensity = min(1.0, total_pheromone / max_pheromone)
                        
                        # Different colors for different pheromone types
                        if edge in self.food_pheromone_map and self.food_pheromone_map[edge] > self.pheromone_map.get(edge, 0):
                            color = 'orange'  # Food paths
                            alpha = 0.4 + intensity * 0.6
                        else:
                            color = 'purple'  # Regular exploration paths
                            alpha = 0.2 + intensity * 0.4
                            
                        linewidth = 0.5 + intensity * 3
                        
                        self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                                      c=color, linewidth=linewidth, alpha=alpha)
        
        # Highlight stable network edges
        if final or self.step_count == self.max_steps: # Also highlight on final frame of animation
            for edge in self.stable_network_edges:
                node1, node2 = edge
                pos1 = self.positions[node1]
                pos2 = self.positions[node2]
                
                self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                             c='red', linewidth=4, alpha=0.8)
        
        # Plot all nodes
        for i in range(self.num_nodes):
            pos = self.positions[i]
            
            if i in self.source_nodes:
                # Source nodes - large red circles
                self.ax.scatter(*pos, c='red', s=300, alpha=0.9, 
                                 edgecolors='darkred', linewidth=2, marker='s')
                self.ax.text(pos[0], pos[1], f'S{i}', fontsize=10, fontweight='bold', 
                             ha='center', va='center', color='white')
            elif i in self.food_nodes:
                # Food nodes - large green circles
                self.ax.scatter(*pos, c='green', s=250, alpha=0.9, 
                                 edgecolors='darkgreen', linewidth=2, marker='o')
                self.ax.text(pos[0], pos[1], f'F{i}', fontsize=9, fontweight='bold', 
                             ha='center', va='center', color='white')
            else:
                # Regular nodes - small blue circles
                self.ax.scatter(*pos, c='lightblue', s=80, alpha=0.7, 
                                 edgecolors='blue', linewidth=1)
                self.ax.text(pos[0], pos[1], str(i), fontsize=7, 
                             ha='center', va='center')
        
        # Plot sample of agents grouped by source
        source_groups = {}
        for agent in self.agents:
            if agent.alive and agent.energy > 0:
                if agent.source_id not in source_groups:
                    source_groups[agent.source_id] = []
                source_groups[agent.source_id].append(agent.position)
        
        colors = ['red', 'blue', 'purple', 'brown', 'pink', 'gray']
        for source_id, positions in source_groups.items():
            if positions:
                positions = np.array(positions)
                # Show sample of agents to avoid clutter
                sample_size = min(15, len(positions))
                indices = np.random.choice(len(positions), sample_size, replace=False)
                sample_positions = positions[indices]
                
                for pos in sample_positions:
                    self.ax.scatter(*pos, c=colors[source_id % len(colors)], s=25, 
                                     alpha=0.8, marker='o', edgecolors='black', linewidth=0.5)
        
        # Set plot properties
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_aspect('equal')
        
        # Title with information
        active_agents = sum(1 for a in self.agents if a.alive and a.energy > 0)
        title = f"Slime Mold Food Network - Step {self.step_count}" # Use self.step_count
        if final:
            title += f" - FINAL: Stable Network ({len(self.stable_network_edges)} edges)"
        else:
            title += f" - Active: {active_agents}, Paths: {len(self.discovered_edges)}, Stable: {len(self.stable_network_edges)}"
            
        self.ax.set_title(title, fontsize=12)
        
        # Set axis limits with margin
        all_positions = np.array(list(self.positions.values()))
        margin = 3
        self.ax.set_xlim(all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin)
        self.ax.set_ylim(all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Source Nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Food Nodes'),
            plt.Line2D([0], [0], color='orange', linewidth=3, label='Food Paths'),
            plt.Line2D([0], [0], color='purple', linewidth=2, label='Exploration Paths'),
        ]
        if final or self.step_count == self.max_steps: # Ensure stable network is shown if simulation reached end
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=4, label='Stable Network'))
            
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        plt.draw() # Ensure the plot is redrawn

def main():
    """Main function to run the animated simulation"""
    # Create and run animated food network simulation
    sim = SlimeMoldFoodNetwork(num_nodes=26, num_sources=None, num_food=None, agents_per_source=40)
    sim.animate_simulation(max_steps=350) # Keep max_steps, but we'll control frames manually
    
    print("\nSimulation Summary:")
    print(f"- Total nodes: {sim.num_nodes}")
    print(f"- Source nodes: {sim.num_sources} - {sim.source_nodes}")
    print(f"- Food nodes: {sim.num_food} - {sim.food_nodes}")
    print(f"- Agents per source: {sim.agents_per_source}")
    print(f"- Total agents: {len(sim.agents)}")
    print(f"- Total paths discovered: {len(sim.discovered_edges)}")
    print(f"- Stable network edges: {len(sim.stable_network_edges)}")

if __name__ == "__main__":
    main()