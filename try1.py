import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import random
from collections import defaultdict, deque
import math

class NetworkNode:
    """A node in the slime mold network"""
    
    def __init__(self, x, y, node_id, generation=0):
        self.x = x
        self.y = y
        self.id = node_id
        self.generation = generation  # How far from source
        self.connections = []  # Connected nodes
        self.resource_level = 10.0  # Amount of resources/nutrients
        self.growth_energy = 10.0  # Energy available for growth
        self.is_source = False
        self.is_food_connected = False
        self.last_growth_time = 0
        self.activity_level = 1.0  # For reinforcement/withering
        
    def add_connection(self, other_node, strength=1.0):
        """Add bidirectional connection between nodes"""
        if other_node not in [conn['node'] for conn in self.connections]:
            self.connections.append({'node': other_node, 'strength': strength})
            other_node.connections.append({'node': self, 'strength': strength})

class FoodSource:
    """Food source that attracts slime mold growth"""
    
    def __init__(self, x, y, nutrition_value=100):
        self.x = x
        self.y = y
        self.nutrition_value = nutrition_value
        self.max_nutrition = nutrition_value
        self.radius = 50
        self.is_connected = False
        self.connected_nodes = []

class SlimeMoldNetwork:
    """Main slime mold network simulation"""
    
    def __init__(self, width=800, height=600, num_sources=2):
        self.width = width
        self.height = height
        
        # Network structure
        self.nodes = {}  # node_id -> NetworkNode
        self.node_counter = 0
        self.source_nodes = []
        
        # Food sources
        self.food_sources = []
        self.max_food_sources = 8
        self.food_spawn_rate = 0.05
        
        # Growth parameters
        self.growth_rate = 5.0  # nodes per step
        self.branch_probability = 0.7
        self.exploration_vs_exploitation = 0.6  # 0=pure exploitation, 1=pure exploration
        self.max_growth_distance = 75
        self.min_growth_distance = 25
        
        # Network optimization
        self.reinforcement_rate = 0.1
        self.decay_rate = 0.02
        self.min_activity_threshold = 0.1
        
        # Visualization
        self.fig = None
        self.ax = None
        self.time_step = 0
        
        # Initialize network
        self.initialize_network(num_sources)
        
    def initialize_network(self, num_sources):
        """Initialize the network with source nodes"""
        self.nodes = {}
        self.node_counter = 0
        self.source_nodes = []
        
        for i in range(num_sources):
            # Place sources with some spacing
            if num_sources == 1:
                x, y = self.width/2, self.height/2
            else:
                angle = 2 * math.pi * i / num_sources
                radius = min(self.width, self.height) * 0.2
                x = self.width/2 + radius * math.cos(angle)
                y = self.height/2 + radius * math.sin(angle)
            
            source_node = NetworkNode(x, y, self.node_counter, generation=0)
            source_node.is_source = True
            source_node.resource_level = 100.0  # Sources have high resources
            source_node.growth_energy = 5.0
            
            self.nodes[self.node_counter] = source_node
            self.source_nodes.append(source_node)
            self.node_counter += 1
    
    def spawn_food(self):
        """Randomly spawn food sources"""
        if len(self.food_sources) < self.max_food_sources and random.random() < self.food_spawn_rate:
            # Avoid spawning too close to existing food or sources
            attempts = 0
            while attempts < 10:
                x = random.uniform(50, self.width - 50)
                y = random.uniform(50, self.height - 50)
                
                # Check distance from existing food and sources
                too_close = False
                min_distance = 80
                
                for food in self.food_sources:
                    if math.sqrt((x - food.x)**2 + (y - food.y)**2) < min_distance:
                        too_close = True
                        break
                
                for source in self.source_nodes:
                    if math.sqrt((x - source.x)**2 + (y - source.y)**2) < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    self.food_sources.append(FoodSource(x, y))
                    break
                
                attempts += 1
    
    def calculate_food_attraction(self, node):
        """Calculate attraction vector toward nearest food"""
        if not self.food_sources:
            return 0, 0, 0  # No attraction
        
        total_attraction_x = 0
        total_attraction_y = 0
        max_attraction = 0
        
        for food in self.food_sources:
            if food.is_connected:
                continue  # Skip already connected food
                
            dx = food.x - node.x
            dy = food.y - node.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > 0:
                # Attraction decreases with distance but has longer range
                attraction_strength = food.nutrition_value / (1 + distance/50)
                
                # Only consider food within detection range
                detection_range = 150
                if distance < detection_range:
                    total_attraction_x += attraction_strength * dx / distance
                    total_attraction_y += attraction_strength * dy / distance
                    max_attraction = max(max_attraction, attraction_strength)
        
        return total_attraction_x, total_attraction_y, max_attraction
    
    def find_growth_direction(self, node):
        """Determine growth direction balancing exploration and exploitation"""
        # Get food attraction
        food_x, food_y, food_strength = self.calculate_food_attraction(node)
        
        # Random exploration component
        random_angle = random.uniform(0, 2 * math.pi)
        explore_x = math.cos(random_angle)
        explore_y = math.sin(random_angle)
        
        # Avoid overcrowded areas (local repulsion)
        repulsion_x, repulsion_y = 0, 0
        for other_node in self.nodes.values():
            if other_node.id != node.id:
                dx = node.x - other_node.x
                dy = node.y - other_node.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < 30 and distance > 0:  # Too close
                    repulsion_strength = 1.0 / (distance + 1)
                    repulsion_x += repulsion_strength * dx / distance
                    repulsion_y += repulsion_strength * dy / distance
        
        # Combine forces based on exploration vs exploitation balance
        final_x = (self.exploration_vs_exploitation * explore_x + 
                  (1 - self.exploration_vs_exploitation) * food_x + 
                  repulsion_x)
        final_y = (self.exploration_vs_exploitation * explore_y + 
                  (1 - self.exploration_vs_exploitation) * food_y + 
                  repulsion_y)
        
        # Normalize
        magnitude = math.sqrt(final_x**2 + final_y**2)
        if magnitude > 0:
            final_x /= magnitude
            final_y /= magnitude
        
        return final_x, final_y, food_strength > 0
    
    def grow_network(self):
        """Grow the network by adding new nodes"""
        growth_candidates = []
        
        # Find nodes that can grow
        for node in self.nodes.values():
            if (node.growth_energy > 1.0 and 
                node.activity_level > self.min_activity_threshold and
                self.time_step - node.last_growth_time > 5):
                growth_candidates.append(node)
        
        # Sort by growth energy (prioritize well-resourced nodes)
        growth_candidates.sort(key=lambda n: n.growth_energy, reverse=True)
        
        # Limit growth rate
        max_growths = max(1, int(self.growth_rate))
        growth_candidates = growth_candidates[:max_growths]
        
        for node in growth_candidates:
            # Determine if this node should branch
            should_branch = (random.random() < self.branch_probability and 
                           len(node.connections) < 3)
            
            num_new_nodes = 2 if should_branch else 1
            
            for _ in range(num_new_nodes):
                # Find growth direction
                dir_x, dir_y, has_food_attraction = self.find_growth_direction(node)
                
                # Add some randomness to direction
                angle_noise = random.uniform(-0.5, 0.5)
                current_angle = math.atan2(dir_y, dir_x) + angle_noise
                
                # Determine growth distance
                base_distance = self.min_growth_distance
                if has_food_attraction:
                    growth_distance = base_distance + random.uniform(5, 15)
                else:
                    growth_distance = base_distance + random.uniform(0, 10)
                
                # Calculate new position
                new_x = node.x + growth_distance * math.cos(current_angle)
                new_y = node.y + growth_distance * math.sin(current_angle)
                
                # Boundary constraints
                new_x = max(10, min(self.width - 10, new_x))
                new_y = max(10, min(self.height - 10, new_y))
                
                # Create new node
                new_node = NetworkNode(new_x, new_y, self.node_counter, 
                                     generation=node.generation + 1)
                new_node.resource_level = node.resource_level * 0.8
                new_node.growth_energy = node.growth_energy * 0.7
                new_node.activity_level = 0.8
                
                # Add to network
                self.nodes[self.node_counter] = new_node
                node.add_connection(new_node)
                self.node_counter += 1
                
                # Reduce parent's growth energy
                node.growth_energy *= 0.8
                node.last_growth_time = self.time_step
    
    def check_food_connections(self):
        """Check if any nodes have reached food sources"""
        for food in self.food_sources:
            if food.is_connected:
                continue
                
            for node in self.nodes.values():
                distance = math.sqrt((node.x - food.x)**2 + (node.y - food.y)**2)
                if distance < food.radius:
                    # Connection made!
                    food.is_connected = True
                    food.connected_nodes.append(node)
                    node.is_food_connected = True
                    
                    # Trace back path to source and reinforce
                    self.reinforce_path_to_food(node)
                    break
    
    def reinforce_path_to_food(self, food_node):
        """Reinforce the path from food back to sources"""
        # Use BFS to find path back to source
        queue = deque([(food_node, [])])
        visited = set()
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node.id in visited:
                continue
            visited.add(current_node.id)
            
            new_path = path + [current_node]
            
            if current_node.is_source:
                # Found path to source, reinforce it
                for node in new_path:
                    node.activity_level = min(2.0, node.activity_level + 0.5)
                    node.resource_level = min(5.0, node.resource_level + 1.0)
                    node.growth_energy = min(3.0, node.growth_energy + 0.5)
                
                # Strengthen connections in the path
                for i in range(len(new_path) - 1):
                    node1, node2 = new_path[i], new_path[i + 1]
                    for conn in node1.connections:
                        if conn['node'] == node2:
                            conn['strength'] = min(3.0, conn['strength'] + 0.3)
                break
            
            # Add neighbors to queue
            for conn in current_node.connections:
                if conn['node'].id not in visited:
                    queue.append((conn['node'], new_path))
    
    def update_network_resources(self):
        """Update resource flow and network optimization"""
        # Resource flow from sources
        for source in self.source_nodes:
            source.resource_level = 10.0  # Sources maintain high resources
            source.growth_energy = min(5.0, source.growth_energy + 0.2)
        
        # Flow resources through network (simplified)
        for node in self.nodes.values():
            if not node.is_source:
                # Get resources from connected nodes
                total_incoming = 0
                for conn in node.connections:
                    connected_node = conn['node']
                    flow_rate = conn['strength'] * 0.1
                    resource_flow = connected_node.resource_level * flow_rate
                    total_incoming += resource_flow
                
                # Update resources
                node.resource_level = min(3.0, node.resource_level * 0.95 + total_incoming)
                node.growth_energy = min(2.0, node.resource_level * 0.5)
        
        # Decay unused branches
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            if not node.is_source:
                node.activity_level *= (1 - self.decay_rate)
                
                # Mark for removal if too weak
                if node.activity_level < self.min_activity_threshold:
                    nodes_to_remove.append(node_id)
        
        # Remove weak nodes
        for node_id in nodes_to_remove:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Remove connections
                for conn in node.connections:
                    other_node = conn['node']
                    other_node.connections = [c for c in other_node.connections 
                                            if c['node'] != node]
                del self.nodes[node_id]
    
    def update(self):
        """Single simulation step"""
        self.time_step += 1
        
        self.spawn_food()
        self.grow_network()
        self.check_food_connections()
        self.update_network_resources()
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')
        self.ax.set_title('Biological Slime Mold Network Growth', 
                         fontsize=16, color='lime')
        
        # Remove axes for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
    def animate_frame(self, frame):
        """Animation function"""
        if frame > 0:
            self.update()
        
        # Clear plot
        self.ax.clear()
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('black')
        
        # Draw network connections
        for node in self.nodes.values():
            for conn in node.connections:
                other_node = conn['node']
                
                # Line thickness based on connection strength and activity
                thickness = conn['strength'] * node.activity_level * 2
                thickness = max(0.5, min(4.0, thickness))
                
                # Color based on activity and food connection
                if node.is_food_connected or other_node.is_food_connected:
                    color = 'gold'
                    alpha = 0.9
                elif node.activity_level > 0.7:
                    color = 'lime'
                    alpha = 0.8
                else:
                    color = 'cyan'
                    alpha = 0.4 + node.activity_level * 0.4
                
                self.ax.plot([node.x, other_node.x], [node.y, other_node.y],
                           color=color, linewidth=thickness, alpha=alpha)
        
        # Draw nodes
        for node in self.nodes.values():
            if node.is_source:
                # Source nodes are larger and brighter
                self.ax.scatter(node.x, node.y, c='yellow', s=120, 
                              alpha=0.9, edgecolors='orange', linewidth=2, zorder=10)
            elif node.is_food_connected:
                # Food-connected nodes are golden
                size = 30 + node.activity_level * 20
                self.ax.scatter(node.x, node.y, c='gold', s=size, 
                              alpha=0.8, edgecolors='yellow', linewidth=1, zorder=8)
            else:
                # Regular nodes
                size = 10 + node.activity_level * 15
                color_intensity = node.activity_level
                self.ax.scatter(node.x, node.y, c='lime', s=size,
                              alpha=0.3 + color_intensity * 0.5, 
                              edgecolors='white', linewidth=0.5, zorder=5)
        
        # Draw food sources
        for food in self.food_sources:
            color = 'orange' if food.is_connected else 'red'
            alpha = 0.7 if food.is_connected else 1.0
            
            # Food glow
            glow_circle = Circle((food.x, food.y), food.radius * 2,
                               color=color, alpha=alpha*0.3, zorder=1)
            self.ax.add_patch(glow_circle)
            
            # Food core
            core_circle = Circle((food.x, food.y), food.radius,
                               color=color, alpha=alpha, zorder=3)
            self.ax.add_patch(core_circle)
        
        # Update title with stats
        num_nodes = len(self.nodes)
        num_connections = sum(len(node.connections) for node in self.nodes.values()) // 2
        connected_food = sum(1 for food in self.food_sources if food.is_connected)
        
        title = f'Slime Mold Network - Nodes: {num_nodes}, Connections: {num_connections}, '
        title += f'Food Found: {connected_food}/{len(self.food_sources)}, Time: {self.time_step}'
        
        self.ax.set_title(title, fontsize=12, color='lime')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
    
    def run_simulation(self, max_frames=2000, interval=100, save_video=False):
        """Run the animated simulation"""
        self.setup_visualization()
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=max_frames,
            interval=interval, repeat=False, blit=False
        )
        
        # Save video if requested
        if save_video:
            print("Saving video... This may take a while.")
            writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='SlimeMold'), 
                                          bitrate=1800)
            anim.save('slime_mold_network.mp4', writer=writer)
            print("Video saved as 'slime_mold_network.mp4'")
        
        plt.tight_layout()
        plt.show()
        return anim

# Example usage
if __name__ == "__main__":
    print("Biological Slime Mold Network Simulation")
    print("=" * 50)
    print("Watch as the slime mold grows from source points,")
    print("explores the environment, finds food, and optimizes its network!")
    print()
    
    # Create simulation
    # You can adjust parameters:
    # - num_sources: 1-3 starting points
    # - width/height: environment size
    sim = SlimeMoldNetwork(width=800, height=600, num_sources=2)
    
    # Adjust growth parameters for different behaviors:
    sim.growth_rate = 1.5  # How fast the network grows
    sim.branch_probability = 0.25  # How often it branches
    sim.exploration_vs_exploitation = 0.7  # 0=go straight to food, 1=explore randomly
    
    # Run simulation
    print("Starting simulation... Close the window to stop.")
    sim.run_simulation(max_frames=1500, interval=80, save_video=False)
    
    # For video export, uncomment:
    # sim.run_simulation(max_frames=1000, interval=50, save_video=True)