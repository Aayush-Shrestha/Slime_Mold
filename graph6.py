import pygame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import deque
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

class CSVSlimeSimulation:
    def __init__(self, csv_file=None, width=800, height=800, agent_count=8000):
        # Configuration
        self.WIDTH = width
        self.HEIGHT = height
        self.AGENT_COUNT = agent_count
        self.TRAIL_DECAY = 0.88
        self.FOOD_TRAIL_DECAY = 0.95
        self.SENSOR_DISTANCE = 10
        self.SENSOR_ANGLE = math.pi / 4
        self.TURN_ANGLE = math.pi / 6
        self.STEP_SIZE = 1.8
        self.MEMORY_LENGTH = 50
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("CSV-Based Slime Simulation")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()
        
        # Initialize simulation components
        self.exploration_trail = np.zeros((self.WIDTH, self.HEIGHT))
        self.food_trail = np.zeros((self.WIDTH, self.HEIGHT))
        self.obstacle_mask = np.ones((self.WIDTH, self.HEIGHT))  # 1 = passable, 0 = blocked
        
        # CSV-related attributes
        self.csv_data = None
        self.grid_x = None
        self.grid_y = None
        self.grid_values = None
        self.x_coords = None
        self.y_coords = None
        
        # Simulation elements
        self.sources = []
        self.destinations = []
        self.agents = []
        
        # Load CSV if provided
        if csv_file:
            self.load_csv(csv_file)
        else:
            self.create_demo_environment()
    
    def load_csv(self, csv_file):
        """Load CSV file and create environment"""
        try:
            print(f"Loading CSV file: {csv_file}")
            self.csv_data = pd.read_csv(csv_file)
            
            # Ensure required columns exist
            required_cols = ['X', 'Y', 'VALUE']
            for col in required_cols:
                if col not in self.csv_data.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV")
            
            # Extract coordinates and values
            self.x_coords = self.csv_data['X'].values
            self.y_coords = self.csv_data['Y'].values
            values = self.csv_data['VALUE'].values
            
            # Normalize coordinates to screen dimensions
            self.normalize_coordinates()
            
            # Create obstacle mask from CSV data
            self.create_obstacle_mask_from_csv(values)
            
            # Generate sources and destinations
            self.generate_sources_and_destinations()
            
            print(f"Loaded {len(self.csv_data)} points from CSV")
            print(f"Generated {len(self.sources)} sources and {len(self.destinations)} destinations")
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Creating demo environment instead")
            self.create_demo_environment()
    
    def normalize_coordinates(self):
        """Normalize CSV coordinates to screen dimensions"""
        # Get bounds
        x_min, x_max = self.x_coords.min(), self.x_coords.max()
        y_min, y_max = self.y_coords.min(), self.y_coords.max()
        
        # Normalize to screen size with padding
        padding = 50
        self.x_coords = ((self.x_coords - x_min) / (x_max - x_min) * 
                        (self.WIDTH - 2 * padding) + padding).astype(int)
        self.y_coords = ((self.y_coords - y_min) / (y_max - y_min) * 
                        (self.HEIGHT - 2 * padding) + padding).astype(int)
        
        # Clip to screen bounds
        self.x_coords = np.clip(self.x_coords, 0, self.WIDTH - 1)
        self.y_coords = np.clip(self.y_coords, 0, self.HEIGHT - 1)
    
    def create_obstacle_mask_from_csv(self, values):
        """Create obstacle mask from CSV values"""
        # Reset obstacle mask
        self.obstacle_mask = np.ones((self.WIDTH, self.HEIGHT))
        
        # Create a more detailed grid representation
        # First, mark all CSV points
        for i in range(len(self.x_coords)):
            x, y = int(self.x_coords[i]), int(self.y_coords[i])
            if values[i] == 1:  # Obstacle
                # Create small obstacle regions around each obstacle point
                self.add_obstacle_region(x, y, radius=1)
        
        # Apply additional smoothing/interpolation if needed
        self.interpolate_obstacles(values)
    
    def add_obstacle_region(self, center_x, center_y, radius=3):
        """Add a small obstacle region around a point"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT:
                        self.obstacle_mask[x, y] = 0.0
    
    def interpolate_obstacles(self, values):
        """Interpolate obstacles for smoother boundaries"""
        # Create a sparse grid for interpolation
        try:
            from scipy.interpolate import griddata
            
            # Create regular grid
            xi = np.linspace(0, self.WIDTH-1, self.WIDTH//4)
            yi = np.linspace(0, self.HEIGHT-1, self.HEIGHT//4)
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate values
            points = np.column_stack((self.x_coords, self.y_coords))
            interpolated = griddata(points, values, (xi_grid, yi_grid), 
                                  method='nearest', fill_value=0)
            
            # Apply interpolated values to mask
            for i in range(len(xi)):
                for j in range(len(yi)):
                    if interpolated[j, i] > 0.5:  # Threshold for obstacles
                        x_start = int(i * 4)
                        x_end = min(self.WIDTH, int((i+1) * 4))
                        y_start = int(j * 4)
                        y_end = min(self.HEIGHT, int((j+1) * 4))
                        self.obstacle_mask[x_start:x_end, y_start:y_end] = 0.0
                        
        except ImportError:
            print("SciPy not available for interpolation, using basic method")
    
    def create_demo_environment(self):
        """Create a demo environment for testing"""
        print("Creating demo environment")
        
        # Reset obstacle mask
        self.obstacle_mask = np.ones((self.WIDTH, self.HEIGHT))
        
        # Add some demo obstacles
        self.add_demo_obstacles()
        
        # Generate sources and destinations
        self.generate_sources_and_destinations()
    
    def add_demo_obstacles(self):
        """Add demo obstacles for testing"""
        # Add some circular obstacles
        obstacles = [
            (200, 150, 40),
            (500, 300, 35),
            (350, 450, 30),
            (600, 200, 25),
            (150, 400, 45)
        ]
        
        for x, y, radius in obstacles:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        ox, oy = x + dx, y + dy
                        if 0 <= ox < self.WIDTH and 0 <= oy < self.HEIGHT:
                            self.obstacle_mask[ox, oy] = 0.0
    
    def generate_sources_and_destinations(self):
        """Generate 3 sources and 3 destinations in walkable areas"""
        self.sources = []
        self.destinations = []
        
        # Find all walkable positions
        walkable_positions = []
        for x in range(0, self.WIDTH, 10):  # Sample every 10 pixels for efficiency
            for y in range(0, self.HEIGHT, 10):
                if self.obstacle_mask[x, y] > 0.5:
                    walkable_positions.append((x, y))
        
        if len(walkable_positions) < 6:
            print("Warning: Not enough walkable positions found")
            return
        
        # Randomly select positions, ensuring they're spread out
        selected_positions = []
        for _ in range(6):  # 3 sources + 3 destinations
            attempts = 0
            while attempts < 100:
                pos = random.choice(walkable_positions)
                # Check if position is far enough from already selected positions
                if all(math.sqrt((pos[0] - sp[0])**2 + (pos[1] - sp[1])**2) > 80 
                       for sp in selected_positions):
                    selected_positions.append(pos)
                    break
                attempts += 1
            
            if attempts >= 100 and len(selected_positions) < 6:
                # If we can't find well-separated positions, just add any walkable position
                pos = random.choice(walkable_positions)
                selected_positions.append(pos)
        
        # Split into sources and destinations
        self.sources = selected_positions[:3]
        self.destinations = selected_positions[3:6]
        
        print(f"Sources: {self.sources}")
        print(f"Destinations: {self.destinations}")
    
    def create_agents(self):
        """Create slime agents starting from sources"""
        self.agents = []
        agents_per_source = self.AGENT_COUNT // len(self.sources)
        
        for source in self.sources:
            for _ in range(agents_per_source):
                self.agents.append(SlimeAgent(source[0], source[1], self.destinations))
    
    def update_simulation(self):
        """Update one step of the simulation"""
        # Move all agents
        for agent in self.agents:
            agent.move(self.obstacle_mask, self.exploration_trail, 
                      self.food_trail, self.WIDTH, self.HEIGHT)
        
        # Update trail decay
        self.exploration_trail *= self.TRAIL_DECAY
        self.food_trail *= self.FOOD_TRAIL_DECAY
    
    def draw(self, show_mode="both"):
        """Draw the simulation"""
        self.screen.fill((0, 0, 0))
        
        # Draw trail maps
        if show_mode in ["exploration", "both"]:
            explore_surface = pygame.surfarray.make_surface(
                (np.clip(self.exploration_trail * 128, 0, 255)).astype(np.uint8)
            )
            explore_surface.set_colorkey((0, 0, 0))
            self.screen.blit(explore_surface, (0, 0))
        
        if show_mode in ["food", "both"]:
            food_surface = pygame.surfarray.make_surface(
                (np.clip(self.food_trail * 64, 0, 255)).astype(np.uint8)
            )
            # Tint food trails yellow/orange
            food_array = pygame.surfarray.array3d(food_surface)
            food_array[:, :, 0] = np.clip(food_array[:, :, 0] * 1.5, 0, 255)
            food_array[:, :, 1] = np.clip(food_array[:, :, 1] * 1.2, 0, 255)
            food_surface = pygame.surfarray.make_surface(food_array)
            food_surface.set_colorkey((0, 0, 0))
            self.screen.blit(food_surface, (0, 0))
        
        # Draw obstacles
        obstacle_surface = pygame.surfarray.make_surface(
            ((1 - self.obstacle_mask) * 100).astype(np.uint8)
        )
        obstacle_surface.set_colorkey((0, 0, 0))
        self.screen.blit(obstacle_surface, (0, 0))
        
        # Draw sources (red circles)
        for sx, sy in self.sources:
            pygame.draw.circle(self.screen, (255, 50, 50), (int(sx), int(sy)), 12)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), 15, 2)
        
        # Draw destinations (green circles)
        for dx, dy in self.destinations:
            pygame.draw.circle(self.screen, (50, 255, 50), (int(dx), int(dy)), 10)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(dx), int(dy)), 13, 2)
        
        # Draw UI elements
        self.draw_ui()
    
    def draw_ui(self):
        """Draw user interface elements"""
        # Draw statistics
        exploring = sum(1 for a in self.agents if a.mode == "exploring")
        returning = sum(1 for a in self.agents if a.mode == "returning")
        depositing = sum(1 for a in self.agents if a.mode == "depositing")
        
        stats_text = self.font.render(
            f"Exploring: {exploring} | Returning: {returning} | Depositing: {depositing}",
            True, (255, 255, 255)
        )
        self.screen.blit(stats_text, (10, self.HEIGHT - 30))
        
        # Draw instructions
        if self.csv_data is None:
            instruction_text = self.font.render(
                "Demo Mode - Press R to reset, Q to quit, SPACE to pause",
                True, (200, 200, 200)
            )
            self.screen.blit(instruction_text, (10, 10))
    
    def run(self):
        """Main simulation loop"""
        # Create agents
        self.create_agents()
        
        running = True
        paused = False
        show_mode = "both"
        
        print("Simulation started. Press SPACE to pause, R to reset, Q to quit")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif event.key == pygame.K_r:
                        self.reset_simulation()
                        print("Simulation reset")
                    elif event.key == pygame.K_v:
                        modes = ["both", "exploration", "food"]
                        current_idx = modes.index(show_mode)
                        show_mode = modes[(current_idx + 1) % len(modes)]
                        print(f"View mode: {show_mode}")
            
            if not paused:
                self.update_simulation()
            
            self.draw(show_mode)
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.exploration_trail.fill(0)
        self.food_trail.fill(0)
        self.generate_sources_and_destinations()
        self.create_agents()
    
    def export_paths(self, filename="slime_paths.csv"):
        """Export the current path network to CSV"""
        try:
            path_data = []
            for x in range(0, self.WIDTH, 5):  # Sample every 5 pixels
                for y in range(0, self.HEIGHT, 5):
                    trail_strength = self.food_trail[x, y] + self.exploration_trail[x, y]
                    if trail_strength > 0.1:  # Only export significant paths
                        path_data.append({
                            'X': x,
                            'Y': y,
                            'TRAIL_STRENGTH': trail_strength,
                            'IS_FOOD_TRAIL': self.food_trail[x, y] > self.exploration_trail[x, y]
                        })
            
            df = pd.DataFrame(path_data)
            df.to_csv(filename, index=False)
            print(f"Exported {len(path_data)} path points to {filename}")
        except Exception as e:
            print(f"Error exporting paths: {e}")


class SlimeAgent:
    def __init__(self, start_x, start_y, destinations):
        self.start_x = start_x
        self.start_y = start_y
        self.x = start_x + random.uniform(-20, 20)
        self.y = start_y + random.uniform(-20, 20)
        self.angle = random.uniform(0, 2 * math.pi)
        self.destinations = destinations
        self.mode = "exploring"  # "exploring", "returning", "depositing"
        self.target_destination = None
        self.memory = deque(maxlen=50)
        self.return_timer = 0
        
        # Enhanced parameters
        self.sensor_distance = 10
        self.sensor_angle = math.pi / 4
        self.turn_angle = math.pi / 6
        self.step_size = 1.8
    
    def move(self, obstacle_mask, exploration_trail, food_trail, width, height):
        """Move the agent based on current mode"""
        if self.mode == "exploring":
            self._explore_move(obstacle_mask, exploration_trail, food_trail, width, height)
        elif self.mode == "returning":
            self._return_move(obstacle_mask, food_trail, width, height)
        elif self.mode == "depositing":
            self._deposit_move(food_trail)
        
        # Keep agents on screen
        self.x = max(0, min(width - 1, self.x))
        self.y = max(0, min(height - 1, self.y))
        
        # Record position in memory
        self.memory.append((self.x, self.y))
        
        # Check for destinations if exploring
        if self.mode == "exploring":
            self._check_for_destinations()
    
    def _explore_move(self, obstacle_mask, exploration_trail, food_trail, width, height):
        """Exploration movement with trail following"""
        self._sense_and_turn(exploration_trail, food_trail, obstacle_mask, width, height)
        
        dx = math.cos(self.angle) * self.step_size
        dy = math.sin(self.angle) * self.step_size
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check if new position is blocked
        if (0 <= new_x < width and 0 <= new_y < height and 
            obstacle_mask[int(new_x)][int(new_y)] > 0.5):
            self.x = new_x
            self.y = new_y
        else:
            # Bounce off obstacles
            self.angle += random.uniform(-math.pi/2, math.pi/2)
        
        # Leave exploration trail
        if (0 <= int(self.x) < width and 0 <= int(self.y) < height and
            obstacle_mask[int(self.x)][int(self.y)] > 0.5):
            exploration_trail[int(self.x)][int(self.y)] += 0.8
    
    def _return_move(self, obstacle_mask, food_trail, width, height):
        """Return to source"""
        dx = self.start_x - self.x
        dy = self.start_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 25:  # Reached source
            self.mode = "depositing"
            self.return_timer = 40
            return
        
        # Move toward source
        if distance > 0:
            target_angle = math.atan2(dy, dx)
            new_x = self.x + math.cos(target_angle) * self.step_size
            new_y = self.y + math.sin(target_angle) * self.step_size
            
            # Check if path is blocked
            if (0 <= new_x < width and 0 <= new_y < height and 
                obstacle_mask[int(new_x)][int(new_y)] > 0.5):
                self.angle = target_angle
                self.x = new_x
                self.y = new_y
            else:
                # Navigate around obstacle
                self.angle += random.uniform(-math.pi/4, math.pi/4)
                new_x = self.x + math.cos(self.angle) * self.step_size
                new_y = self.y + math.sin(self.angle) * self.step_size
                if (0 <= new_x < width and 0 <= new_y < height and 
                    obstacle_mask[int(new_x)][int(new_y)] > 0.5):
                    self.x = new_x
                    self.y = new_y
        
        # Leave strong food trail
        if (0 <= int(self.x) < width and 0 <= int(self.y) < height and
            obstacle_mask[int(self.x)][int(self.y)] > 0.5):
            food_trail[int(self.x)][int(self.y)] += 4.0
    
    def _deposit_move(self, food_trail):
        """Deposit phase at source"""
        self.return_timer -= 1
        if self.return_timer <= 0:
            self.mode = "exploring"
            self.target_destination = None
            self.angle = random.uniform(0, 2 * math.pi)
        
        # Strong deposition at source
        food_trail[int(self.x)][int(self.y)] += 6.0
    
    def _sense_and_turn(self, exploration_trail, food_trail, obstacle_mask, width, height):
        """Sense trails and adjust direction"""
        center_explore = self._sample_sensor(0, exploration_trail, obstacle_mask, width, height)
        left_explore = self._sample_sensor(-self.sensor_angle, exploration_trail, obstacle_mask, width, height)
        right_explore = self._sample_sensor(self.sensor_angle, exploration_trail, obstacle_mask, width, height)
        
        center_food = self._sample_sensor(0, food_trail, obstacle_mask, width, height) * 2.5
        left_food = self._sample_sensor(-self.sensor_angle, food_trail, obstacle_mask, width, height) * 2.5
        right_food = self._sample_sensor(self.sensor_angle, food_trail, obstacle_mask, width, height) * 2.5
        
        center = center_explore + center_food
        left = left_explore + left_food
        right = right_explore + right_food
        
        if center > left and center > right:
            return
        elif left > right:
            self.angle -= self.turn_angle
        elif right > left:
            self.angle += self.turn_angle
        else:
            self.angle += random.uniform(-self.turn_angle, self.turn_angle)
    
    def _sample_sensor(self, angle_offset, trail_map, obstacle_mask, width, height):
        """Sample trail strength at sensor position"""
        angle = self.angle + angle_offset
        sx = int((self.x + math.cos(angle) * self.sensor_distance))
        sy = int((self.y + math.sin(angle) * self.sensor_distance))
        
        if (0 <= sx < width and 0 <= sy < height and 
            obstacle_mask[sx][sy] > 0.5):
            return trail_map[sx][sy]
        return 0
    
    def _check_for_destinations(self):
        """Check if agent has reached any destination"""
        for dest in self.destinations:
            distance = math.sqrt((self.x - dest[0])**2 + (self.y - dest[1])**2)
            if distance < 25:
                self.mode = "returning"
                self.target_destination = dest
                self._reinforce_memory_trail()
                break
    
    def _reinforce_memory_trail(self):
        """Strengthen trail along memory path"""
        # This would be implemented to reinforce the food_trail
        # based on the agent's memory of the path taken
        pass


def create_sample_csv(filename="sample_environment.csv"):
    """Create a sample CSV file for testing"""
    print(f"Creating sample CSV file: {filename}")
    
    # Create a grid with some obstacles
    size = 100
    data = []
    
    for i in range(size):
        for j in range(size):
            x = i * 8  # Scale up coordinates
            y = j * 6
            
            # Create some obstacle patterns
            value = 0  # Default walkable
            
            # Add some circular obstacles
            if ((x - 200)**2 + (y - 150)**2) < 1600:  # Circle 1
                value = 1
            elif ((x - 500)**2 + (y - 300)**2) < 2500:  # Circle 2
                value = 1
            elif ((x - 350)**2 + (y - 450)**2) < 1200:  # Circle 3
                value = 1
            # Add some rectangular obstacles
            elif 150 <= x <= 250 and 400 <= y <= 500:
                value = 1
            elif 600 <= x <= 750 and 100 <= y <= 200:
                value = 1
            
            data.append({'X': x, 'Y': y, 'VALUE': value})
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created sample CSV with {len(data)} points")
    return filename


def main():
    """Main function to run the simulation"""
    print("CSV-Based Slime Simulation")
    print("=" * 40)
    
    # Ask user for CSV file
    csv_file = input("Enter CSV file path (or press Enter for demo mode): ").strip()
    
    if not csv_file:
        print("Running in demo mode")
        sim = CSVSlimeSimulation()
    else:
        try:
            sim = CSVSlimeSimulation(csv_file)
        except FileNotFoundError:
            print(f"File {csv_file} not found. Creating sample CSV...")
            sample_file = create_sample_csv()
            sim = CSVSlimeSimulation(sample_file)
    
    print("\nControls:")
    print("SPACE - Pause/Resume")
    print("R - Reset simulation")
    print("V - Toggle view mode")
    print("Q - Quit")
    print("\nStarting simulation...")
    
    sim.run()


if __name__ == "__main__":
    main()