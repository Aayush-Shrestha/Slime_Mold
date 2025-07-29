import pygame
import random
import math
import numpy as np

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 12000
TRAIL_DECAY = 0.99
SENSOR_DISTANCE = 10
SENSOR_ANGLE = math.pi / 4
TURN_ANGLE = math.pi / 8
STEP_SIZE = 1.2
SPAWN_MODE = "network"
NODE_ATTRACTION = 20  # Strength of node attraction
PATH_REINFORCEMENT = 5  # How much agents reinforce successful paths

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime Mold Network Optimization")
font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 18)
clock = pygame.time.Clock()

trail_map = np.zeros((WIDTH, HEIGHT))

# Network nodes (like cities in the Japanese railway experiment)
network_nodes = [
    (150, 150, "Tokyo"),
    (650, 150, "Sendai"), 
    (400, 200, "Nagoya"),
    (200, 350, "Osaka"),
    (600, 380, "Kyoto"),
    (450, 480, "Hiroshima"),
    (300, 500, "Fukuoka"),
    (500, 300, "Kobe")
]

# Track which nodes agents have visited for path optimization
agent_paths = {}

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

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def find_nearest_node(x, y):
    min_dist = float('inf')
    nearest = None
    for i, (nx, ny, name) in enumerate(network_nodes):
        dist = distance((x, y), (nx, ny))
        if dist < min_dist:
            min_dist = dist
            nearest = i
    return nearest, min_dist

# --- Enhanced Slime Agent for Network Optimization ---
class NetworkSlimeAgent:
    def __init__(self, x, y, angle=None):
        self.x = x
        self.y = y
        self.angle = angle if angle is not None else random.uniform(0, 2 * math.pi)
        self.target_node = None
        self.visited_nodes = []
        self.path_quality = 0
        self.id = random.randint(0, 1000000)
        
    def move(self):
        dx = math.cos(self.angle) * STEP_SIZE
        dy = math.sin(self.angle) * STEP_SIZE
        self.x = (self.x + dx) % WIDTH
        self.y = (self.y + dy) % HEIGHT
        
        # Deposit trail with path quality influence
        trail_strength = 1 + (self.path_quality * 0.5)
        trail_map[int(self.x)][int(self.y)] += trail_strength
        
        # Check if near a node
        nearest_node, dist = find_nearest_node(self.x, self.y)
        if dist < 25 and nearest_node not in self.visited_nodes:
            self.visited_nodes.append(nearest_node)
            self.path_quality += 1
            # Choose a new target
            self._choose_target()

    def sense(self):
        center = self._sample_sensor(0)
        left = self._sample_sensor(-SENSOR_ANGLE)
        right = self._sample_sensor(SENSOR_ANGLE)
        
        # Add node attraction to sensing
        node_influence = self._calculate_node_attraction()
        
        # Combine trail following with node attraction
        center += node_influence[0]
        left += node_influence[1] 
        right += node_influence[2]

        if center > left and center > right:
            return
        elif left > right:
            self.angle -= TURN_ANGLE
        elif right > left:
            self.angle += TURN_ANGLE
        else:
            self.angle += random.uniform(-TURN_ANGLE/2, TURN_ANGLE/2)

    def _sample_sensor(self, angle_offset):
        angle = self.angle + angle_offset
        sx = int((self.x + math.cos(angle) * SENSOR_DISTANCE) % WIDTH)
        sy = int((self.y + math.sin(angle) * SENSOR_DISTANCE) % HEIGHT)
        return trail_map[sx][sy]
    
    def _calculate_node_attraction(self):
        """Calculate attraction to nodes in three directions (center, left, right)"""
        attractions = [0, 0, 0]  # center, left, right
        
        if self.target_node is not None:
            tx, ty, _ = network_nodes[self.target_node]
            
            # Calculate angles to target for each sensor direction
            angles = [
                self.angle,  # center
                self.angle - SENSOR_ANGLE,  # left
                self.angle + SENSOR_ANGLE   # right  
            ]
            
            for i, sensor_angle in enumerate(angles):
                # Point in sensor direction
                sensor_x = self.x + math.cos(sensor_angle) * SENSOR_DISTANCE
                sensor_y = self.y + math.sin(sensor_angle) * SENSOR_DISTANCE
                
                # Calculate if this direction is towards target
                target_angle = math.atan2(ty - self.y, tx - self.x)
                angle_diff = abs(sensor_angle - target_angle)
                angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Handle wrap-around
                
                # Stronger attraction when pointing towards target
                if angle_diff < math.pi/3:  # Within 60 degrees
                    dist_to_target = distance((self.x, self.y), (tx, ty))
                    attraction = NODE_ATTRACTION / (1 + dist_to_target/100)
                    attractions[i] += attraction
                    
        return attractions
    
    def _choose_target(self):
        """Choose a target node to move towards"""
        unvisited = [i for i in range(len(network_nodes)) if i not in self.visited_nodes]
        if unvisited:
            # Choose closest unvisited node with some randomness
            if random.random() < 0.7:  # 70% choose closest
                closest = min(unvisited, key=lambda i: distance(
                    (self.x, self.y), (network_nodes[i][0], network_nodes[i][1])
                ))
                self.target_node = closest
            else:  # 30% choose random unvisited
                self.target_node = random.choice(unvisited)
        else:
            # All nodes visited, choose random node
            self.target_node = random.randint(0, len(network_nodes)-1)

def create_agents(mode="network"):
    agents = []
    if mode == "network":
        # Spawn agents near network nodes
        for _ in range(AGENT_COUNT):
            if random.random() < 0.8:  # 80% spawn near nodes
                node_x, node_y, _ = random.choice(network_nodes)
                x = node_x + random.uniform(-30, 30)
                y = node_y + random.uniform(-30, 30)
            else:  # 20% spawn randomly
                x = random.uniform(0, WIDTH)
                y = random.uniform(0, HEIGHT)
            
            agent = NetworkSlimeAgent(x, y)
            agent._choose_target()
            agents.append(agent)
    elif mode == "grid":
        for _ in range(AGENT_COUNT):
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, HEIGHT)
            agents.append(NetworkSlimeAgent(x, y))
    
    return agents

agents = create_agents(SPAWN_MODE)

# --- Main loop ---
running = True
frame_count = 0

while running:
    screen.fill((0, 0, 0))
    click = False
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
            if event.button == 1:  # Left click = add new node
                node_name = f"Node{len(network_nodes)+1}"
                network_nodes.append((mouse_pos[0], mouse_pos[1], node_name))
                # Reassign targets for some agents
                for agent in random.sample(agents, min(100, len(agents))):
                    agent._choose_target()

    # Agent behavior
    for agent in agents:
        agent.sense()
        agent.move()

    # Update trails with decay
    trail_map *= TRAIL_DECAY

    # Reinforce paths between connected nodes
    if frame_count % 10 == 0:  # Every 10 frames
        for i, (x1, y1, _) in enumerate(network_nodes):
            for j, (x2, y2, _) in enumerate(network_nodes[i+1:], i+1):
                # Add slight reinforcement along direct paths between nodes
                steps = int(distance((x1, y1), (x2, y2)) / 2)
                if steps > 0:
                    for step in range(steps):
                        t = step / steps
                        px = int(x1 + t * (x2 - x1))
                        py = int(y1 + t * (y2 - y1))
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            # Check if there's already a strong trail nearby
                            local_trail = trail_map[max(0, px-2):min(WIDTH, px+3), 
                                                   max(0, py-2):min(HEIGHT, py+3)]
                            if np.max(local_trail) > 5:  # If agents are using this area
                                trail_map[px][py] += 0.5

    # Draw trail map with enhanced visualization
    trail_normalized = np.clip(trail_map, 0, 255)
    
    # Create colored trail visualization
    trail_colored = np.zeros((WIDTH, HEIGHT, 3))
    trail_colored[:, :, 0] = trail_normalized * 0.3  # Red component
    trail_colored[:, :, 1] = trail_normalized * 0.8  # Green component (main)
    trail_colored[:, :, 2] = trail_normalized * 0.1  # Blue component
    
    surface = pygame.surfarray.make_surface(trail_colored.astype(np.uint8))
    surface.set_alpha(200)
    screen.blit(surface, (0, 0))

    # Draw network nodes
    for i, (nx, ny, name) in enumerate(network_nodes):
        # Node circle
        pygame.draw.circle(screen, (255, 200, 50), (int(nx), int(ny)), 12)
        pygame.draw.circle(screen, (200, 150, 0), (int(nx), int(ny)), 12, 2)
        
        # Node label
        label = small_font.render(name, True, (255, 255, 255))
        screen.blit(label, (nx - 20, ny - 30))

    # Draw some agent positions as small dots
    if frame_count % 5 == 0:  # Update every 5 frames for performance
        sample_agents = random.sample(agents, min(50, len(agents)))
        for agent in sample_agents:
            pygame.draw.circle(screen, (255, 255, 100), (int(agent.x), int(agent.y)), 1)

    # Draw buttons and info
    clear_btn = pygame.Rect(10, 10, 100, 30)
    reset_btn = pygame.Rect(120, 10, 120, 30)
    mode_btn = pygame.Rect(250, 10, 160, 30)
    
    action1 = draw_button(clear_btn, "Clear Trails", mouse_pos, click, "clear")
    action2 = draw_button(reset_btn, "Reset Network", mouse_pos, click, "reset")
    mode_text = "Mode: " + ("Network" if SPAWN_MODE == "network" else "Grid")
    action3 = draw_button(mode_btn, mode_text, mouse_pos, click, "toggle_mode")

    # Info display
    info_text = f"Nodes: {len(network_nodes)} | Agents: {len(agents)} | Click to add nodes"
    info_surface = small_font.render(info_text, True, (200, 200, 200))
    screen.blit(info_surface, (10, HEIGHT - 25))

    # Handle button actions
    if action1 == "clear":
        trail_map.fill(0)
    if action2 == "reset":
        network_nodes = [
            (150, 150, "Tokyo"),
            (650, 150, "Sendai"), 
            (400, 200, "Nagoya"),
            (200, 350, "Osaka"),
            (600, 380, "Kyoto"),
            (450, 480, "Hiroshima"),
            (300, 500, "Fukuoka"),
            (500, 300, "Kobe")
        ]
        agents = create_agents(SPAWN_MODE)
        trail_map.fill(0)
    if action3 == "toggle_mode":
        SPAWN_MODE = "network" if SPAWN_MODE == "grid" else "grid"
        agents = create_agents(SPAWN_MODE)
        trail_map.fill(0)

    frame_count += 1
    pygame.display.flip()
    clock.tick(60)

pygame.quit()