import pygame
import random
import math
import numpy as np
from collections import deque

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 2000  # Reduced from 4000 for better performance
TRAIL_DECAY = 0.85
FOOD_TRAIL_DECAY = 0.95  # Slower decay for food trails
EVACUATION_TRAIL_DECAY = 0.90  # Emergency evacuation trails
SENSOR_DISTANCE = 8
SENSOR_ANGLE = math.pi / 4
TURN_ANGLE = math.pi / 6
STEP_SIZE = 1.5
MEMORY_LENGTH = 40  # How many steps agents remember
SAFETY_NODE_COUNT = 6
SAFETY_DETECTION_RADIUS = 20
FIRE_DETECTION_RADIUS = 25  # Agents can detect fire from further away
COLONY_CENTERS = 3

# Fire system configuration
FIRE_SPREAD_RATE = 0.5  # Pixels per frame (1 pixel every 2 frames)
WIND_DIRECTION = math.pi / 4  # Wind blowing northeast
WIND_STRENGTH = 1.5  # Multiplier for fire spread in wind direction
FIRE_INTENSITY_DECAY = 0.995  # How fast fire burns out
SMOKE_SPREAD_RATE = 2.0  # Smoke spreads faster than fire
FIRE_UPDATE_SKIP = 2  # Update fire every N frames for performance

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime Mold Evacuation Route Optimization")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# Trail maps: exploration, evacuation, and fire/smoke
exploration_trail = np.zeros((WIDTH, HEIGHT))
evacuation_trail = np.zeros((WIDTH, HEIGHT))  # Emergency evacuation routes
fire_map = np.zeros((WIDTH, HEIGHT))  # Fire intensity
smoke_map = np.zeros((WIDTH, HEIGHT))  # Smoke density

# Obstacle system
obstacle_mask = np.ones((WIDTH, HEIGHT))  # 1 = passable, 0 = blocked
obstacles = []  # List to store obstacle info for drawing

# Initialize safety nodes and colonies
safety_nodes = []
colony_centers = []
disaster_active = False
fire_update_counter = 0  # Counter for fire update optimization

def generate_safety_nodes_and_colonies():
    global safety_nodes, colony_centers
    safety_nodes = []
    colony_centers = []
    
    # Generate safety nodes with better spacing
    attempts = 0
    while len(safety_nodes) < SAFETY_NODE_COUNT and attempts < 100:
        new_safety = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
        # Ensure minimum distance between safety nodes
        if all(math.sqrt((new_safety[0] - sx)**2 + (new_safety[1] - sy)**2) > 80 
               for sx, sy in safety_nodes):
            if obstacle_mask[int(new_safety[0])][int(new_safety[1])] > 0.5:
                safety_nodes.append(new_safety)
        attempts += 1
    
    # Generate colony centers
    attempts = 0
    while len(colony_centers) < COLONY_CENTERS and attempts < 100:
        new_colony = (random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100))
        # Ensure colonies are not too close to safety nodes or each other
        if (all(math.sqrt((new_colony[0] - sx)**2 + (new_colony[1] - sy)**2) > 60 
                for sx, sy in safety_nodes) and
            all(math.sqrt((new_colony[0] - cx)**2 + (new_colony[1] - cy)**2) > 100
                for cx, cy in colony_centers)):
            if obstacle_mask[int(new_colony[0])][int(new_colony[1])] > 0.5:
                colony_centers.append(new_colony)
        attempts += 1

def add_circular_obstacle(center_x, center_y, radius, obstacle_type="wall"):
    """Add a circular obstacle"""
    global obstacles, obstacle_mask
    
    obstacles.append({
        "type": "circle",
        "center": (center_x, center_y),
        "radius": radius,
        "obstacle_type": obstacle_type
    })
    
    # Update obstacle mask
    for x in range(max(0, center_x - radius), min(WIDTH, center_x + radius + 1)):
        for y in range(max(0, center_y - radius), min(HEIGHT, center_y + radius + 1)):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                obstacle_mask[x][y] = 0.0

def add_rectangular_obstacle(x, y, width, height, obstacle_type="wall"):
    """Add a rectangular obstacle"""
    global obstacles, obstacle_mask
    
    obstacles.append({
        "type": "rectangle",
        "pos": (x, y),
        "size": (width, height),
        "obstacle_type": obstacle_type
    })
    
    # Update obstacle mask
    for ox in range(max(0, x), min(WIDTH, x + width)):
        for oy in range(max(0, y), min(HEIGHT, y + height)):
            obstacle_mask[ox][oy] = 0.0

def add_polygon_obstacle(points, obstacle_type="wall"):
    """Add a polygon obstacle (list of (x,y) points)"""
    global obstacles, obstacle_mask
    
    obstacles.append({
        "type": "polygon",
        "points": points,
        "obstacle_type": obstacle_type
    })
    
    # Simple point-in-polygon check for obstacle mask
    min_x = max(0, int(min(p[0] for p in points)))
    max_x = min(WIDTH, int(max(p[0] for p in points)) + 1)
    min_y = max(0, int(min(p[1] for p in points)))
    max_y = min(HEIGHT, int(max(p[1] for p in points)) + 1)
    
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            if point_in_polygon(x, y, points):
                obstacle_mask[x][y] = 0.0

def point_in_polygon(x, y, points):
    """Check if point is inside polygon using ray casting"""
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def clear_obstacles():
    """Remove all obstacles"""
    global obstacles, obstacle_mask
    obstacles.clear()
    obstacle_mask.fill(1.0)

def start_wildfire():
    """Start wildfire at random locations"""
    global disaster_active, fire_map
    disaster_active = True
    
    # Start 2-4 fires at random locations
    fire_count = random.randint(2, 4)
    for _ in range(fire_count):
        fx = random.randint(50, WIDTH - 50)
        fy = random.randint(50, HEIGHT - 50)
        
        # Ensure fire doesn't start in obstacles or too close to colonies
        if (obstacle_mask[fx][fy] > 0.5 and 
            all(math.sqrt((fx - cx)**2 + (fy - cy)**2) > 80 for cx, cy in colony_centers)):
            fire_map[fx][fy] = 255.0  # Maximum fire intensity

def update_fire_system():
    """Update fire and smoke spread"""
    global fire_map, smoke_map
    
    if not disaster_active:
        return
    
    new_fire_map = fire_map.copy()
    new_smoke_map = smoke_map.copy()
    
    # Fire spread
    for x in range(1, WIDTH - 1):
        for y in range(1, HEIGHT - 1):
            if fire_map[x][y] > 50:  # Active fire threshold
                # Calculate spread in all directions with wind influence
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                            
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                            # Check if area is passable and not already burning
                            if obstacle_mask[nx][ny] > 0.5 and fire_map[nx][ny] < 10:
                                # Calculate wind influence
                                spread_angle = math.atan2(dy, dx)
                                wind_factor = 1.0 + WIND_STRENGTH * math.cos(spread_angle - WIND_DIRECTION)
                                wind_factor = max(0.3, wind_factor)  # Minimum spread rate
                                
                                # Lakes act as fire barriers
                                barrier_factor = 1.0
                                for obs in obstacles:
                                    if obs["obstacle_type"] == "lake":
                                        if obs["type"] == "circle":
                                            cx, cy = obs["center"]
                                            dist = math.sqrt((nx - cx)**2 + (ny - cy)**2)
                                            if dist < obs["radius"] + 20:  # Barrier zone around lake
                                                barrier_factor *= 0.1
                                
                                spread_rate = FIRE_SPREAD_RATE * wind_factor * barrier_factor
                                new_fire_map[nx][ny] += fire_map[x][y] * spread_rate * 0.01
    
    # Fire decay and smoke generation
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # Fire burns out over time
            new_fire_map[x][y] *= FIRE_INTENSITY_DECAY
            
            # Generate smoke from fire
            if fire_map[x][y] > 20:
                new_smoke_map[x][y] += fire_map[x][y] * 0.5
            
            # Smoke spreads and dissipates
            if smoke_map[x][y] > 1:
                # Smoke spreads to adjacent cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and obstacle_mask[nx][ny] > 0.5:
                            new_smoke_map[nx][ny] += smoke_map[x][y] * 0.05
                
                # Smoke dissipates
                new_smoke_map[x][y] *= 0.98
    
    fire_map = np.clip(new_fire_map, 0, 255)
    smoke_map = np.clip(new_smoke_map, 0, 100)

def is_safety_node_available(node_pos):
    """Check if safety node is not consumed by fire"""
    x, y = int(node_pos[0]), int(node_pos[1])
    return fire_map[x][y] < 30  # Node unavailable if fire intensity > 30

# Add some default obstacles
add_circular_obstacle(200, 150, 40, "mountain")
add_rectangular_obstacle(500, 300, 80, 60, "building")
add_polygon_obstacle([(350, 450), (400, 420), (450, 480), (380, 500)], "lake")

generate_safety_nodes_and_colonies()

# --- Enhanced Slime Agent ---
class EnhancedSlimeAgent:
    def __init__(self, colony_x, colony_y):
        self.colony_x = colony_x
        self.colony_y = colony_y
        self.x = colony_x + random.uniform(-30, 30)
        self.y = colony_y + random.uniform(-30, 30)
        self.angle = random.uniform(0, 2 * math.pi)
        self.mode = "exploring"  # "exploring", "returning", "evacuating", "depositing"
        self.has_resources = False
        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.return_timer = 0
        self.panic_level = 0  # How panicked the agent is
        
    def move(self):
        # Detect fire and update panic level
        self._detect_fire()
        
        if self.mode == "exploring":
            self._explore_move()
        elif self.mode == "returning":
            self._return_move()
        elif self.mode == "evacuating":
            self._evacuate_move()
        elif self.mode == "depositing":
            self._deposit_move()
            
        # Keep agents on screen
        self.x = max(0, min(WIDTH - 1, self.x))
        self.y = max(0, min(HEIGHT - 1, self.y))
        
        # Record position in memory
        self.memory.append((self.x, self.y))
        
        # Check for safety nodes if exploring
        if self.mode == "exploring":
            self._check_for_safety_nodes()
            
    def _detect_fire(self):
        """Detect fire and smoke, adjust behavior accordingly"""
        fire_detected = False
        smoke_detected = False
        
        # Check fire in detection radius
        for dx in range(-FIRE_DETECTION_RADIUS, FIRE_DETECTION_RADIUS + 1):
            for dy in range(-FIRE_DETECTION_RADIUS, FIRE_DETECTION_RADIUS + 1):
                fx, fy = int(self.x + dx), int(self.y + dy)
                if 0 <= fx < WIDTH and 0 <= fy < HEIGHT:
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= FIRE_DETECTION_RADIUS:
                        if fire_map[fx][fy] > 10:
                            fire_detected = True
                            # Closer fire = more panic
                            panic_increase = (FIRE_DETECTION_RADIUS - distance) / FIRE_DETECTION_RADIUS
                            self.panic_level = min(10, self.panic_level + panic_increase * 0.5)
                        if smoke_map[fx][fy] > 5:
                            smoke_detected = True
        
        # Switch to evacuation mode if fire detected
        if fire_detected and self.mode != "evacuating":
            self.mode = "evacuating"
            self.has_resources = False  # Drop everything and evacuate
        
        # Gradual panic decay when no fire
        if not fire_detected:
            self.panic_level = max(0, self.panic_level - 0.1)
            
    def _explore_move(self):
        # Normal exploration behavior with fire avoidance
        self._sense_and_turn()
        dx = math.cos(self.angle) * STEP_SIZE
        dy = math.sin(self.angle) * STEP_SIZE
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check if new position is safe (no obstacles, low fire/smoke)
        if (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT and 
            obstacle_mask[int(new_x)][int(new_y)] > 0.5 and
            fire_map[int(new_x)][int(new_y)] < 5 and
            smoke_map[int(new_x)][int(new_y)] < 15):
            self.x = new_x
            self.y = new_y
        else:
            # Avoid dangerous areas
            self.angle += random.uniform(-math.pi/2, math.pi/2)
        
        # Leave exploration trail
        if (0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT and
            obstacle_mask[int(self.x)][int(self.y)] > 0.5):
            exploration_trail[int(self.x)][int(self.y)] += 0.5
            
    def _return_move(self):
        # Move toward colony center
        dx = self.colony_x - self.x
        dy = self.colony_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 20:  # Reached colony
            self.mode = "depositing"
            self.return_timer = 30
            return
            
        # Move toward colony with fire avoidance
        if distance > 0:
            target_angle = math.atan2(dy, dx)
            new_x = self.x + math.cos(target_angle) * STEP_SIZE
            new_y = self.y + math.sin(target_angle) * STEP_SIZE
            
            # Check if path is safe
            if (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT and 
                obstacle_mask[int(new_x)][int(new_y)] > 0.5 and
                fire_map[int(new_x)][int(new_y)] < 10):
                self.angle = target_angle
                self.x = new_x
                self.y = new_y
            else:
                # Navigate around obstacles and fire
                self.angle += random.uniform(-math.pi/4, math.pi/4)
                new_x = self.x + math.cos(self.angle) * STEP_SIZE
                new_y = self.y + math.sin(self.angle) * STEP_SIZE
                if (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT and 
                    obstacle_mask[int(new_x)][int(new_y)] > 0.5 and
                    fire_map[int(new_x)][int(new_y)] < 10):
                    self.x = new_x
                    self.y = new_y
            
            # Leave resource trail while returning
            if (0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT and
                obstacle_mask[int(self.x)][int(self.y)] > 0.5):
                evacuation_trail[int(self.x)][int(self.y)] += 3.0
                
    def _evacuate_move(self):
        """Emergency evacuation to nearest colony"""
        # Find nearest colony
        min_distance = float('inf')
        target_colony = None
        
        for cx, cy in colony_centers:
            distance = math.sqrt((self.x - cx)**2 + (self.y - cy)**2)
            if distance < min_distance:
                min_distance = distance
                target_colony = (cx, cy)
        
        if target_colony and min_distance < 20:
            # Reached safety
            self.mode = "depositing"
            self.return_timer = 60  # Stay longer during evacuation
            self.panic_level = max(0, self.panic_level - 2)
            return
        
        if target_colony:
            dx = target_colony[0] - self.x
            dy = target_colony[1] - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Move toward nearest colony with increased urgency
                target_angle = math.atan2(dy, dx)
                speed_multiplier = 1.0 + self.panic_level * 0.1  # Panic makes agents move faster
                step = STEP_SIZE * speed_multiplier
                
                new_x = self.x + math.cos(target_angle) * step
                new_y = self.y + math.sin(target_angle) * step
                
                # Strongly avoid fire during evacuation
                if (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT and 
                    obstacle_mask[int(new_x)][int(new_y)] > 0.5 and
                    fire_map[int(new_x)][int(new_y)] < 5):
                    self.angle = target_angle
                    self.x = new_x
                    self.y = new_y
                else:
                    # Try alternative routes
                    for alt_angle in [target_angle + math.pi/3, target_angle - math.pi/3]:
                        alt_x = self.x + math.cos(alt_angle) * step
                        alt_y = self.y + math.sin(alt_angle) * step
                        if (0 <= alt_x < WIDTH and 0 <= alt_y < HEIGHT and 
                            obstacle_mask[int(alt_x)][int(alt_y)] > 0.5 and
                            fire_map[int(alt_x)][int(alt_y)] < 10):
                            self.x = alt_x
                            self.y = alt_y
                            self.angle = alt_angle
                            break
                
                # Leave strong evacuation trail
                if (0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT and
                    obstacle_mask[int(self.x)][int(self.y)] > 0.5):
                    evacuation_trail[int(self.x)][int(self.y)] += 5.0
                
    def _deposit_move(self):
        # Stay near colony
        self.return_timer -= 1
        if self.return_timer <= 0:
            if disaster_active and self.panic_level > 2:
                self.mode = "evacuating"  # Continue evacuating if still panicked
            else:
                self.mode = "exploring"
                self.has_resources = False
                self.angle = random.uniform(0, 2 * math.pi)
            
        # Strong deposition at colony
        if 0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT:
            evacuation_trail[int(self.x)][int(self.y)] += 5.0
            
    def _sense_and_turn(self):
        # Enhanced sensing with fire avoidance
        center_explore = self._sample_sensor(0, exploration_trail)
        left_explore = self._sample_sensor(-SENSOR_ANGLE, exploration_trail)
        right_explore = self._sample_sensor(SENSOR_ANGLE, exploration_trail)
        
        center_evac = self._sample_sensor(0, evacuation_trail) * 3  # Prioritize evacuation routes
        left_evac = self._sample_sensor(-SENSOR_ANGLE, evacuation_trail) * 3
        right_evac = self._sample_sensor(SENSOR_ANGLE, evacuation_trail) * 3
        
        # Fire avoidance - negative weights for fire/smoke
        center_fire = -self._sample_fire_sensor(0) * 5
        left_fire = -self._sample_fire_sensor(-SENSOR_ANGLE) * 5
        right_fire = -self._sample_fire_sensor(SENSOR_ANGLE) * 5
        
        center = center_explore + center_evac + center_fire
        left = left_explore + left_evac + left_fire
        right = right_explore + right_evac + right_fire
        
        if center > left and center > right:
            return
        elif left > right:
            self.angle -= TURN_ANGLE
        elif right > left:
            self.angle += TURN_ANGLE
        else:
            self.angle += random.uniform(-TURN_ANGLE, TURN_ANGLE)
            
    def _sample_sensor(self, angle_offset, trail_map):
        angle = self.angle + angle_offset
        sx = int((self.x + math.cos(angle) * SENSOR_DISTANCE))
        sy = int((self.y + math.sin(angle) * SENSOR_DISTANCE))
        
        if (0 <= sx < WIDTH and 0 <= sy < HEIGHT and 
            obstacle_mask[sx][sy] > 0.5):
            return trail_map[sx][sy]
        return 0
    
    def _sample_fire_sensor(self, angle_offset):
        """Sample fire and smoke intensity in sensor direction"""
        angle = self.angle + angle_offset
        sx = int((self.x + math.cos(angle) * SENSOR_DISTANCE))
        sy = int((self.y + math.sin(angle) * SENSOR_DISTANCE))
        
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            return fire_map[sx][sy] + smoke_map[sx][sy] * 0.5
        return 0
        
    def _check_for_safety_nodes(self):
        for i, node_pos in enumerate(safety_nodes):
            if not is_safety_node_available(node_pos):
                continue  # Skip nodes consumed by fire
                
            distance = math.sqrt((self.x - node_pos[0])**2 + (self.y - node_pos[1])**2)
            if distance < SAFETY_DETECTION_RADIUS:
                self.has_resources = True
                self.mode = "returning"
                
                # Reinforce the path taken to reach safety node
                self._reinforce_memory_trail()
                break
                
    def _reinforce_memory_trail(self):
        # Strengthen the trail along recently visited positions
        for i, (mx, my) in enumerate(self.memory):
            if 0 <= int(mx) < WIDTH and 0 <= int(my) < HEIGHT:
                strength = (i / len(self.memory)) * 2
                evacuation_trail[int(mx)][int(my)] += strength

# Create agents spawning from colony centers
agents = []
for _ in range(AGENT_COUNT):
    colony = random.choice(colony_centers)
    agents.append(EnhancedSlimeAgent(colony[0], colony[1]))

# --- Buttons ---
button_color = (70, 70, 70)
hover_color = (100, 100, 100)
text_color = (255, 255, 255)

def draw_button(rect, text, mouse_pos, click, action):
    color = hover_color if rect.collidepoint(mouse_pos) else button_color
    pygame.draw.rect(screen, color, rect)
    label = font.render(text, True, text_color)
    screen.blit(label, (rect.x + 5, rect.y + 5))
    if rect.collidepoint(mouse_pos) and click:
        return action
    return None

# --- Main loop ---
running = True
show_mode = "both"  # "exploration", "evacuation", "both"

while running:
    screen.fill((0, 0, 0))
    click = False
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.KEYDOWN:
            # Check for Ctrl+Shift+F to start wildfire
            keys = pygame.key.get_pressed()
            if (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]) and \
               (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) and \
               event.key == pygame.K_f:
                start_wildfire()

    # Update fire system
    update_fire_system()

    # Agent behavior
    for agent in agents:
        agent.move()

    # Update trails with different decay rates
    exploration_trail *= TRAIL_DECAY
    evacuation_trail *= EVACUATION_TRAIL_DECAY

    # Draw fire and smoke first (background)
    if disaster_active:
        # Draw smoke
        smoke_surface = pygame.surfarray.make_surface(
            (np.clip(smoke_map * 2, 0, 255)).astype(np.uint8)
        )
        # Tint smoke gray
        smoke_array = pygame.surfarray.array3d(smoke_surface)
        smoke_surface = pygame.surfarray.make_surface(smoke_array)
        smoke_surface.set_colorkey((0, 0, 0))
        smoke_surface.set_alpha(128)  # Semi-transparent smoke
        screen.blit(smoke_surface, (0, 0))
        
        # Draw fire
        fire_surface = pygame.surfarray.make_surface(
            (np.clip(fire_map, 0, 255)).astype(np.uint8)
        )
        # Tint fire red-orange
        fire_array = pygame.surfarray.array3d(fire_surface)
        fire_array[:, :, 0] = np.clip(fire_array[:, :, 0], 0, 255)  # Red channel
        fire_array[:, :, 1] = np.clip(fire_array[:, :, 1] * 0.6, 0, 255)  # Green channel (orange)
        fire_array[:, :, 2] = np.clip(fire_array[:, :, 2] * 0.1, 0, 255)  # Blue channel (minimal)
        fire_surface = pygame.surfarray.make_surface(fire_array)
        fire_surface.set_colorkey((0, 0, 0))
        screen.blit(fire_surface, (0, 0))

    # Draw trail maps
    if show_mode in ["exploration", "both"]:
        explore_surface = pygame.surfarray.make_surface(
            (np.clip(exploration_trail * 128, 0, 255)).astype(np.uint8)
        )
        explore_surface.set_colorkey((0, 0, 0))
        screen.blit(explore_surface, (0, 0))
        
    if show_mode in ["evacuation", "both"]:
        evac_surface = pygame.surfarray.make_surface(
            (np.clip(evacuation_trail * 64, 0, 255)).astype(np.uint8)
        )
        # Tint evacuation trails blue
        evac_array = pygame.surfarray.array3d(evac_surface)
        evac_array[:, :, 0] = np.clip(evac_array[:, :, 0] * 0.3, 0, 255)  # Less red
        evac_array[:, :, 1] = np.clip(evac_array[:, :, 1] * 0.8, 0, 255)  # Some green
        evac_array[:, :, 2] = np.clip(evac_array[:, :, 2] * 1.5, 0, 255)  # More blue
        evac_surface = pygame.surfarray.make_surface(evac_array)
        evac_surface.set_colorkey((0, 0, 0))
        screen.blit(evac_surface, (0, 0))

    # Draw obstacles
    for obs in obstacles:
        if obs["type"] == "circle":
            cx, cy = obs["center"]
            radius = obs["radius"]
            color = {"mountain": (139, 69, 19), "building": (128, 128, 128), "lake": (0, 100, 200), "wall": (100, 100, 100)}
            obstacle_color = color.get(obs["obstacle_type"], (100, 100, 100))
            pygame.draw.circle(screen, obstacle_color, (int(cx), int(cy)), radius)
            pygame.draw.circle(screen, (200, 200, 200), (int(cx), int(cy)), radius, 2)
            
        elif obs["type"] == "rectangle":
            x, y = obs["pos"]
            width, height = obs["size"]
            color = {"mountain": (139, 69, 19), "building": (128, 128, 128), "lake": (0, 100, 200), "wall": (100, 100, 100)}
            obstacle_color = color.get(obs["obstacle_type"], (100, 100, 100))
            pygame.draw.rect(screen, obstacle_color, (x, y, width, height))
            pygame.draw.rect(screen, (200, 200, 200), (x, y, width, height), 2)
            
        elif obs["type"] == "polygon":
            points = obs["points"]
            color = {"mountain": (139, 69, 19), "building": (128, 128, 128), "lake": (0, 100, 200), "wall": (100, 100, 100)}
            obstacle_color = color.get(obs["obstacle_type"], (100, 100, 100))
            if len(points) >= 3:
                pygame.draw.polygon(screen, obstacle_color, points)
                pygame.draw.polygon(screen, (200, 200, 200), points, 2)

    # Draw safety nodes (available ones in green, unavailable in red)
    for sx, sy in safety_nodes:
        if is_safety_node_available((sx, sy)):
            pygame.draw.circle(screen, (0, 255, 100), (int(sx), int(sy)), 8)
            pygame.draw.circle(screen, (255, 255, 255), (int(sx), int(sy)), 10, 2)
        else:
            pygame.draw.circle(screen, (255, 50, 50), (int(sx), int(sy)), 8)
            pygame.draw.circle(screen, (255, 255, 255), (int(sx), int(sy)), 10, 2)

    # Draw colony centers (safe zones)
    for cx, cy in colony_centers:
        pygame.draw.circle(screen, (100, 100, 255), (int(cx), int(cy)), 15)
        pygame.draw.circle(screen, (255, 255, 255), (int(cx), int(cy)), 18, 3)

    # Draw buttons
    clear_btn = pygame.Rect(10, 10, 100, 30)
    reset_btn = pygame.Rect(120, 10, 100, 30)
    view_btn = pygame.Rect(230, 10, 140, 30)
    obstacle_btn = pygame.Rect(380, 10, 130, 30)
    fire_btn = pygame.Rect(520, 10, 100, 30)
    
    action1 = draw_button(clear_btn, "Clear Trails", mouse_pos, click, "clear")
    action2 = draw_button(reset_btn, "Reset World", mouse_pos, click, "reset")
    view_text = f"View: {show_mode.title()}"
    action3 = draw_button(view_btn, view_text, mouse_pos, click, "toggle_view")
    action4 = draw_button(obstacle_btn, "Clear Obstacles", mouse_pos, click, "clear_obstacles")
    fire_text = "Stop Fire" if disaster_active else "Start Fire"
    action5 = draw_button(fire_btn, fire_text, mouse_pos, click, "toggle_fire")

    if action1 == "clear":
        exploration_trail.fill(0)
        evacuation_trail.fill(0)
    if action2 == "reset":
        exploration_trail.fill(0)
        evacuation_trail.fill(0)
        fire_map.fill(0)
        smoke_map.fill(0)
        disaster_active = False
        generate_safety_nodes_and_colonies()
        agents = []
        for _ in range(AGENT_COUNT):
            colony = random.choice(colony_centers)
            agents.append(EnhancedSlimeAgent(colony[0], colony[1]))
    if action3 == "toggle_view":
        modes = ["both", "exploration", "evacuation"]
        current_idx = modes.index(show_mode)
        show_mode = modes[(current_idx + 1) % len(modes)]
    if action4 == "clear_obstacles":
        clear_obstacles()
        generate_safety_nodes_and_colonies()
    if action5 == "toggle_fire":
        if disaster_active:
            disaster_active = False
            fire_map.fill(0)
            smoke_map.fill(0)
            # Reset agent modes
            for agent in agents:
                if agent.mode == "evacuating":
                    agent.mode = "exploring"
                agent.panic_level = 0
        else:
            start_wildfire()
        
    # Handle mouse clicks for adding obstacles
    keys = pygame.key.get_pressed()
    if click and not any([clear_btn.collidepoint(mouse_pos), reset_btn.collidepoint(mouse_pos), 
                         view_btn.collidepoint(mouse_pos), obstacle_btn.collidepoint(mouse_pos),
                         fire_btn.collidepoint(mouse_pos)]):
        if keys[pygame.K_c]:  # Hold C for circular obstacle
            add_circular_obstacle(mouse_pos[0], mouse_pos[1], 30)
        elif keys[pygame.K_r]:  # Hold R for rectangular obstacle
            add_rectangular_obstacle(mouse_pos[0] - 25, mouse_pos[1] - 25, 50, 50)
        elif keys[pygame.K_l]:  # Hold L for lake (circular, blue)
            add_circular_obstacle(mouse_pos[0], mouse_pos[1], 35, "lake")
        elif keys[pygame.K_m]:  # Hold M for mountain (circular, brown)
            add_circular_obstacle(mouse_pos[0], mouse_pos[1], 40, "mountain")
        elif keys[pygame.K_b]:  # Hold B for building (rectangular, gray)
            add_rectangular_obstacle(mouse_pos[0] - 30, mouse_pos[1] - 20, 60, 40, "building")
        elif keys[pygame.K_f]:  # Hold F to manually start fire at mouse position
            if not disaster_active:
                disaster_active = True
            fire_map[mouse_pos[0]][mouse_pos[1]] = 255.0

    # Draw stats
    exploring = sum(1 for a in agents if a.mode == "exploring")
    returning = sum(1 for a in agents if a.mode == "returning")
    evacuating = sum(1 for a in agents if a.mode == "evacuating")
    depositing = sum(1 for a in agents if a.mode == "depositing")
    
    # Calculate average panic level
    avg_panic = sum(a.panic_level for a in agents) / len(agents) if agents else 0
    
    # Count available safety nodes
    available_nodes = sum(1 for node in safety_nodes if is_safety_node_available(node))
    
    stats_text1 = font.render(f"Exploring: {exploring} | Returning: {returning} | Evacuating: {evacuating} | Depositing: {depositing}", 
                             True, (255, 255, 255))
    stats_text2 = font.render(f"Avg Panic: {avg_panic:.1f} | Available Safety Nodes: {available_nodes}/{len(safety_nodes)} | Fire Active: {disaster_active}", 
                             True, (255, 255, 255))
    
    screen.blit(stats_text1, (10, HEIGHT - 50))
    screen.blit(stats_text2, (10, HEIGHT - 25))
    
    # Draw instructions
    if not disaster_active:
        instr_text = font.render("Press Ctrl+Shift+F to start wildfire | Hold F+Click to ignite manually", True, (200, 200, 200))
        screen.blit(instr_text, (10, HEIGHT - 75))
    else:
        instr_text = font.render("Disaster Active! Agents evacuating to blue colonies. Green=Safe nodes, Red=Burned nodes", True, (255, 200, 200))
        screen.blit(instr_text, (10, HEIGHT - 75))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()