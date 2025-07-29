import pygame
import random
import math
import numpy as np
from collections import deque

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 3000
TRAIL_DECAY = 0.98
FOOD_TRAIL_DECAY = 0.995  # Slower decay for food trails
SENSOR_DISTANCE = 8
SENSOR_ANGLE = math.pi / 4
TURN_ANGLE = math.pi / 6
STEP_SIZE = 1.5
MEMORY_LENGTH = 50  # How many steps agents remember
FOOD_COUNT = 5
FOOD_DETECTION_RADIUS = 10
COLONY_CENTERS = 3

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Slime Mold Network Formation")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# Two trail maps: exploration and food trails
exploration_trail = np.zeros((WIDTH, HEIGHT))
food_trail = np.zeros((WIDTH, HEIGHT))

# Initialize food sources randomly
food_sources = []
colony_centers = []

def generate_food_and_colonies():
    global food_sources, colony_centers
    food_sources = [(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)) 
                   for _ in range(FOOD_COUNT)]
    colony_centers = [(random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)) 
                     for _ in range(COLONY_CENTERS)]

generate_food_and_colonies()

# --- Enhanced Slime Agent ---
class EnhancedSlimeAgent:
    def __init__(self, colony_x, colony_y):
        self.colony_x = colony_x
        self.colony_y = colony_y
        self.x = colony_x + random.uniform(-30, 30)
        self.y = colony_y + random.uniform(-30, 30)
        self.angle = random.uniform(0, 2 * math.pi)
        self.mode = "exploring"  # "exploring", "returning", "depositing"
        self.has_food = False
        self.memory = deque(maxlen=MEMORY_LENGTH)  # Remember recent positions
        self.return_timer = 0
        
    def move(self):
        if self.mode == "exploring":
            self._explore_move()
        elif self.mode == "returning":
            self._return_move()
        elif self.mode == "depositing":
            self._deposit_move()
            
        # Keep agents on screen
        self.x = max(0, min(WIDTH - 1, self.x))
        self.y = max(0, min(HEIGHT - 1, self.y))
        
        # Record position in memory
        self.memory.append((self.x, self.y))
        
        # Check for food if exploring
        if self.mode == "exploring":
            self._check_for_food()
            
    def _explore_move(self):
        # Normal exploration behavior
        self._sense_and_turn()
        dx = math.cos(self.angle) * STEP_SIZE
        dy = math.sin(self.angle) * STEP_SIZE
        self.x += dx
        self.y += dy
        
        # Leave weak exploration trail
        if 0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT:
            exploration_trail[int(self.x)][int(self.y)] += 0.5
            
    def _return_move(self):
        # Move toward colony center
        dx = self.colony_x - self.x
        dy = self.colony_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 20:  # Reached colony
            self.mode = "depositing"
            self.return_timer = 30  # Deposit for 30 frames
            return
            
        # Move toward colony
        if distance > 0:
            self.angle = math.atan2(dy, dx)
            self.x += math.cos(self.angle) * STEP_SIZE
            self.y += math.sin(self.angle) * STEP_SIZE
            
            # Leave strong food trail while returning
            if 0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT:
                food_trail[int(self.x)][int(self.y)] += 3.0
                
    def _deposit_move(self):
        # Stay near colony and deposit nutrients
        self.return_timer -= 1
        if self.return_timer <= 0:
            self.mode = "exploring"
            self.has_food = False
            # Add some randomness to exploration direction
            self.angle = random.uniform(0, 2 * math.pi)
            
        # Strong deposition at colony
        if 0 <= int(self.x) < WIDTH and 0 <= int(self.y) < HEIGHT:
            food_trail[int(self.x)][int(self.y)] += 5.0
            
    def _sense_and_turn(self):
        # Sense both trail types, with preference for food trails
        center_explore = self._sample_sensor(0, exploration_trail)
        left_explore = self._sample_sensor(-SENSOR_ANGLE, exploration_trail)
        right_explore = self._sample_sensor(SENSOR_ANGLE, exploration_trail)
        
        center_food = self._sample_sensor(0, food_trail) * 2  # Weight food trails higher
        left_food = self._sample_sensor(-SENSOR_ANGLE, food_trail) * 2
        right_food = self._sample_sensor(SENSOR_ANGLE, food_trail) * 2
        
        center = center_explore + center_food
        left = left_explore + left_food
        right = right_explore + right_food
        
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
        
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            return trail_map[sx][sy]
        return 0
        
    def _check_for_food(self):
        for i, (fx, fy) in enumerate(food_sources):
            distance = math.sqrt((self.x - fx)**2 + (self.y - fy)**2)
            if distance < FOOD_DETECTION_RADIUS:
                self.has_food = True
                self.mode = "returning"
                
                # Reinforce the path taken to reach food
                self._reinforce_memory_trail()
                
                # Remove consumed food (regenerates later)
                # food_sources.pop(i)
                # # Regenerate food elsewhere after delay
                # if random.random() < 0.1:  # 10% chance to regenerate immediately
                #     new_food = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
                #     food_sources.append(new_food)
                # break
                
    def _reinforce_memory_trail(self):
        # Strengthen the trail along recently visited positions
        for i, (mx, my) in enumerate(self.memory):
            if 0 <= int(mx) < WIDTH and 0 <= int(my) < HEIGHT:
                strength = (i / len(self.memory)) * 2  # Stronger for more recent positions
                food_trail[int(mx)][int(my)] += strength

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
show_mode = "both"  # "exploration", "food", "both"

while running:
    screen.fill((0, 0, 0))
    click = False
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True

    # Agent behavior
    for agent in agents:
        agent.move()

    # Update trails with different decay rates
    exploration_trail *= TRAIL_DECAY
    food_trail *= FOOD_TRAIL_DECAY  # Food trails persist longer

    # Draw trail maps
    if show_mode in ["exploration", "both"]:
        explore_surface = pygame.surfarray.make_surface(
            np.rot90((np.clip(exploration_trail * 128, 0, 255)).astype(np.uint8))
        )
        explore_surface.set_colorkey((0, 0, 0))
        screen.blit(explore_surface, (0, 0))
        
    if show_mode in ["food", "both"]:
        food_surface = pygame.surfarray.make_surface(
            np.rot90((np.clip(food_trail * 64, 0, 255)).astype(np.uint8))
        )
        # Tint food trails yellow/orange
        food_array = pygame.surfarray.array3d(food_surface)
        food_array[:, :, 0] = np.clip(food_array[:, :, 0] * 1.5, 0, 255)  # More red
        food_array[:, :, 1] = np.clip(food_array[:, :, 1] * 1.2, 0, 255)  # Some green
        food_surface = pygame.surfarray.make_surface(food_array)
        food_surface.set_colorkey((0, 0, 0))
        screen.blit(food_surface, (0, 0))

    # Draw food sources
    for fx, fy in food_sources:
        pygame.draw.circle(screen, (0, 255, 100), (int(fx), int(fy)), 6)
        pygame.draw.circle(screen, (255, 255, 255), (int(fx), int(fy)), 8, 2)

    # Draw colony centers
    for cx, cy in colony_centers:
        pygame.draw.circle(screen, (255, 100, 100), (int(cx), int(cy)), 12)
        pygame.draw.circle(screen, (255, 255, 255), (int(cx), int(cy)), 15, 2)

    # Draw buttons
    clear_btn = pygame.Rect(10, 10, 100, 30)
    reset_btn = pygame.Rect(120, 10, 100, 30)
    view_btn = pygame.Rect(230, 10, 120, 30)
    
    action1 = draw_button(clear_btn, "Clear Trails", mouse_pos, click, "clear")
    action2 = draw_button(reset_btn, "Reset World", mouse_pos, click, "reset")
    view_text = f"View: {show_mode.title()}"
    action3 = draw_button(view_btn, view_text, mouse_pos, click, "toggle_view")

    if action1 == "clear":
        exploration_trail.fill(0)
        food_trail.fill(0)
    if action2 == "reset":
        exploration_trail.fill(0)
        food_trail.fill(0)
        generate_food_and_colonies()
        agents = []
        for _ in range(AGENT_COUNT):
            colony = random.choice(colony_centers)
            agents.append(EnhancedSlimeAgent(colony[0], colony[1]))
    if action3 == "toggle_view":
        modes = ["both", "exploration", "food"]
        current_idx = modes.index(show_mode)
        show_mode = modes[(current_idx + 1) % len(modes)]

    # Draw stats
    exploring = sum(1 for a in agents if a.mode == "exploring")
    returning = sum(1 for a in agents if a.mode == "returning")
    depositing = sum(1 for a in agents if a.mode == "depositing")
    
    stats_text = font.render(f"Exploring: {exploring} | Returning: {returning} | Depositing: {depositing}", 
                            True, (255, 255, 255))
    screen.blit(stats_text, (10, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()