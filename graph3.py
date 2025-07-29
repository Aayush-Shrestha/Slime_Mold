import pygame
import random
import math
import numpy as np
from collections import deque

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 2000
TRAIL_DECAY = 0.992
SENSOR_DISTANCE = 12
SENSOR_ANGLE = math.pi / 6
TURN_ANGLE = math.pi / 8
STEP_SIZE = 1.2
MEMORY_LENGTH = 30

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tokyo Railway Slime Mold Simulation")
font = pygame.font.SysFont(None, 20)
small_font = pygame.font.SysFont(None, 16)
clock = pygame.time.Clock()

# Single trail map (like original slime mold)
trail_map = np.zeros((WIDTH, HEIGHT))

# Tokyo and surrounding cities (approximate relative positions)
# Central Tokyo at screen center, surrounding cities scaled to fit
TOKYO_CENTER = (WIDTH // 2, HEIGHT // 2)

# Major cities around Tokyo (scaled and positioned roughly)
cities = [
    # Central Tokyo (start point)
    {"name": "Tokyo", "pos": TOKYO_CENTER, "is_central": True},
    
    # Major surrounding cities
    {"name": "Yokohama", "pos": (WIDTH//2 - 60, HEIGHT//2 + 80), "is_central": False},
    {"name": "Kawasaki", "pos": (WIDTH//2 - 40, HEIGHT//2 + 60), "is_central": False},
    {"name": "Saitama", "pos": (WIDTH//2 - 20, HEIGHT//2 - 80), "is_central": False},
    {"name": "Chiba", "pos": (WIDTH//2 + 90, HEIGHT//2 + 20), "is_central": False},
    {"name": "Hachioji", "pos": (WIDTH//2 - 120, HEIGHT//2 - 20), "is_central": False},
    {"name": "Machida", "pos": (WIDTH//2 - 80, HEIGHT//2 + 40), "is_central": False},
    {"name": "Fujisawa", "pos": (WIDTH//2 - 40, HEIGHT//2 + 120), "is_central": False},
    {"name": "Kashiwa", "pos": (WIDTH//2 + 60, HEIGHT//2 - 60), "is_central": False},
    {"name": "Ichikawa", "pos": (WIDTH//2 + 80, HEIGHT//2 + 60), "is_central": False},
    {"name": "Tachikawa", "pos": (WIDTH//2 - 100, HEIGHT//2 - 60), "is_central": False},
    {"name": "Mitaka", "pos": (WIDTH//2 - 80, HEIGHT//2 - 40), "is_central": False},
    {"name": "Kokubunji", "pos": (WIDTH//2 - 90, HEIGHT//2 - 50), "is_central": False},
    {"name": "Chofu", "pos": (WIDTH//2 - 70, HEIGHT//2 + 20), "is_central": False},
    {"name": "Komae", "pos": (WIDTH//2 - 50, HEIGHT//2 + 30), "is_central": False},
    
    # Additional outer cities
    {"name": "Takasaki", "pos": (WIDTH//2 - 150, HEIGHT//2 - 120), "is_central": False},
    {"name": "Mito", "pos": (WIDTH//2 + 140, HEIGHT//2 - 100), "is_central": False},
    {"name": "Odawara", "pos": (WIDTH//2 - 140, HEIGHT//2 + 100), "is_central": False},
]

# Terrain obstacles (mountains, water bodies that block growth)
obstacles = [
    # Tokyo Bay (simplified)
    {"type": "water", "center": (WIDTH//2 + 120, HEIGHT//2 + 100), "radius": 60},
    {"type": "water", "center": (WIDTH//2 + 80, HEIGHT//2 + 140), "radius": 40},
    
    # Mountain ranges (simplified)
    {"type": "mountain", "center": (WIDTH//2 - 180, HEIGHT//2 - 80), "radius": 50},
    {"type": "mountain", "center": (WIDTH//2 + 160, HEIGHT//2 - 140), "radius": 45},
    {"type": "mountain", "center": (WIDTH//2 - 160, HEIGHT//2 + 140), "radius": 55},
]

# Create obstacle mask
obstacle_mask = np.ones((WIDTH, HEIGHT))  # 1 = passable, 0 = blocked
for obs in obstacles:
    cx, cy = obs["center"]
    radius = obs["radius"]
    for x in range(max(0, cx - radius), min(WIDTH, cx + radius)):
        for y in range(max(0, cy - radius), min(HEIGHT, cy + radius)):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                obstacle_mask[x][y] = 0.1  # Very low permeability

# --- Enhanced Slime Agent for Tokyo Simulation ---
class TokyoSlimeAgent:
    def __init__(self, start_x, start_y):
        self.x = start_x + random.uniform(-15, 15)
        self.y = start_y + random.uniform(-15, 15)
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = 100
        self.last_food_time = 0
        
    def move(self):
        # Sense environment and decide direction
        self._sense_and_turn()
        
        # Move forward
        dx = math.cos(self.angle) * STEP_SIZE
        dy = math.sin(self.angle) * STEP_SIZE
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check if new position is valid (not blocked by obstacles)
        if (0 <= new_x < WIDTH and 0 <= new_y < HEIGHT and 
            obstacle_mask[int(new_x)][int(new_y)] > 0.5):
            self.x = new_x
            self.y = new_y
        else:
            # Bounce off obstacles
            self.angle += random.uniform(-math.pi/2, math.pi/2)
            
        # Deposit trail (stronger near food sources)
        if 0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT:
            food_bonus = self._calculate_food_bonus()
            trail_map[int(self.x)][int(self.y)] += 1.0 + food_bonus
            
        # Lose energy over time, gain energy near food
        self.energy -= 0.1
        if self._near_food():
            self.energy = min(100, self.energy + 2)
            self.last_food_time = pygame.time.get_ticks()
            
        # Die if out of energy
        if self.energy <= 0:
            self._respawn()
            
    def _sense_and_turn(self):
        # Sample trail strength in three directions
        center = self._sample_sensor(0)
        left = self._sample_sensor(-SENSOR_ANGLE)
        right = self._sample_sensor(SENSOR_ANGLE)
        
        # Add food attraction
        food_attraction = self._sense_food_gradient()
        
        if center + food_attraction > left and center + food_attraction > right:
            return  # Continue straight
        elif left > right:
            self.angle -= TURN_ANGLE
        elif right > left:
            self.angle += TURN_ANGLE
        else:
            # Random exploration
            self.angle += random.uniform(-TURN_ANGLE, TURN_ANGLE)
            
    def _sample_sensor(self, angle_offset):
        angle = self.angle + angle_offset
        sx = int(self.x + math.cos(angle) * SENSOR_DISTANCE)
        sy = int(self.y + math.sin(angle) * SENSOR_DISTANCE)
        
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            return trail_map[sx][sy] * obstacle_mask[sx][sy]
        return 0
        
    def _sense_food_gradient(self):
        # Attraction to nearest food source
        min_dist = float('inf')
        for city in cities:
            if not city["is_central"]:  # Don't attract to Tokyo itself
                dx = city["pos"][0] - self.x
                dy = city["pos"][1] - self.y
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
                
        # Stronger attraction when farther from food
        if min_dist > 0:
            return max(0, 20 / min_dist)
        return 0
        
    def _calculate_food_bonus(self):
        # Stronger trails when near food sources
        bonus = 0
        for city in cities:
            dx = city["pos"][0] - self.x
            dy = city["pos"][1] - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 30:  # Near food
                bonus += max(0, (30 - dist) / 30) * 2
        return bonus
        
    def _near_food(self):
        for city in cities:
            dx = city["pos"][0] - self.x
            dy = city["pos"][1] - self.y
            if math.sqrt(dx*dx + dy*dy) < 25:
                return True
        return False
        
    def _respawn(self):
        # Respawn from Tokyo center
        self.x = TOKYO_CENTER[0] + random.uniform(-20, 20)
        self.y = TOKYO_CENTER[1] + random.uniform(-20, 20)
        self.energy = 100
        self.angle = random.uniform(0, 2 * math.pi)

# Create agents starting from Tokyo
agents = []
for _ in range(AGENT_COUNT):
    agents.append(TokyoSlimeAgent(TOKYO_CENTER[0], TOKYO_CENTER[1]))

# --- Buttons ---
button_color = (70, 70, 70)
hover_color = (100, 100, 100)
text_color = (255, 255, 255)

def draw_button(rect, text, mouse_pos, click, action):
    color = hover_color if rect.collidepoint(mouse_pos) else button_color
    pygame.draw.rect(screen, color, rect)
    label = small_font.render(text, True, text_color)
    screen.blit(label, (rect.x + 5, rect.y + 5))
    if rect.collidepoint(mouse_pos) and click:
        return action
    return None

# --- Main loop ---
running = True
show_labels = True
show_obstacles = True
simulation_speed = 1

while running:
    screen.fill((0, 0, 0))
    click = False
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True

    # Run simulation
    for _ in range(simulation_speed):
        for agent in agents:
            agent.move()
        
        # Trail decay
        trail_map *= TRAIL_DECAY

    # Draw trail map
    trail_surface = pygame.surfarray.make_surface(
        (np.clip(trail_map * 80, 0, 255)).astype(np.uint8)
    )
    trail_surface.set_colorkey((0, 0, 0))
    screen.blit(trail_surface, (0, 0))

    # Draw obstacles
    if show_obstacles:
        for obs in obstacles:
            cx, cy = obs["center"]
            color = (0, 100, 200) if obs["type"] == "water" else (100, 80, 60)
            pygame.draw.circle(screen, color, (int(cx), int(cy)), obs["radius"])
            pygame.draw.circle(screen, (150, 150, 150), (int(cx), int(cy)), obs["radius"], 2)

    # Draw cities
    for city in cities:
        x, y = city["pos"]
        if city["is_central"]:
            # Tokyo - larger red circle
            pygame.draw.circle(screen, (255, 50, 50), (int(x), int(y)), 12)
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 15, 3)
        else:
            # Other cities - smaller green circles
            pygame.draw.circle(screen, (50, 255, 100), (int(x), int(y)), 8)
            pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 10, 2)
            
        # City labels
        if show_labels:
            label = small_font.render(city["name"], True, (255, 255, 255))
            screen.blit(label, (int(x) + 15, int(y) - 5))

    # Draw buttons
    clear_btn = pygame.Rect(10, 10, 80, 25)
    labels_btn = pygame.Rect(100, 10, 100, 25)
    obstacles_btn = pygame.Rect(210, 10, 100, 25)
    speed_btn = pygame.Rect(320, 10, 80, 25)
    
    action1 = draw_button(clear_btn, "Clear", mouse_pos, click, "clear")
    labels_text = "Labels: " + ("On" if show_labels else "Off")
    action2 = draw_button(labels_btn, labels_text, mouse_pos, click, "toggle_labels")
    obstacles_text = "Terrain: " + ("On" if show_obstacles else "Off")
    action3 = draw_button(obstacles_btn, obstacles_text, mouse_pos, click, "toggle_obstacles")
    action4 = draw_button(speed_btn, f"Speed: {simulation_speed}x", mouse_pos, click, "speed")

    if action1 == "clear":
        trail_map.fill(0)
    if action2 == "toggle_labels":
        show_labels = not show_labels
    if action3 == "toggle_obstacles":
        show_obstacles = not show_obstacles
    if action4 == "speed":
        simulation_speed = (simulation_speed % 3) + 1

    # Draw info
    info_text = small_font.render("Red = Tokyo (start), Green = Cities (food), Blue/Brown = Water/Mountains", 
                                 True, (200, 200, 200))
    screen.blit(info_text, (10, HEIGHT - 40))
    
    active_agents = sum(1 for a in agents if a.energy > 0)
    stats_text = small_font.render(f"Active agents: {active_agents}/{len(agents)}", 
                                  True, (200, 200, 200))
    screen.blit(stats_text, (10, HEIGHT - 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()