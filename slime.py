import pygame
import random
import math
import numpy as np

# --- Configuration ---
WIDTH, HEIGHT = 800, 600
AGENT_COUNT = 5000
TRAIL_DECAY = 0.95
SENSOR_DISTANCE = 5
SENSOR_ANGLE = math.pi / 4
TURN_ANGLE = math.pi / 6
STEP_SIZE = 1
SPAWN_MODE = "grid"  # "grid" or "few_sources"

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime Mold Simulation")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

trail_map = np.zeros((WIDTH, HEIGHT))
food_sources = [(random.randint(200, WIDTH - 200), random.randint(200, HEIGHT - 200)) for _ in range(7)]

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

# --- Slime Agent ---
class SlimeAgent:
    def __init__(self, x, y, angle=None):
        self.x = x
        self.y = y
        self.angle = angle if angle is not None else random.uniform(0, 2 * math.pi)

    def move(self):
        dx = math.cos(self.angle) * STEP_SIZE
        dy = math.sin(self.angle) * STEP_SIZE
        self.x = (self.x + dx) % WIDTH
        self.y = (self.y + dy) % HEIGHT
        trail_map[int(self.x)][int(self.y)] += 1

    def sense(self):
        center = self._sample_sensor(0)
        left = self._sample_sensor(-SENSOR_ANGLE)
        right = self._sample_sensor(SENSOR_ANGLE)

        if center > left and center > right:
            return
        elif left > right:
            self.angle -= TURN_ANGLE
        elif right > left:
            self.angle += TURN_ANGLE
        else:
            self.angle += random.uniform(-TURN_ANGLE, TURN_ANGLE)

    def _sample_sensor(self, angle_offset):
        angle = self.angle + angle_offset
        sx = int((self.x + math.cos(angle) * SENSOR_DISTANCE) % WIDTH)
        sy = int((self.y + math.sin(angle) * SENSOR_DISTANCE) % HEIGHT)
        return trail_map[sx][sy]

def create_agents(mode="grid"):
    agents = []
    if mode == "grid":
        for _ in range(AGENT_COUNT):
            x = random.uniform(0, WIDTH)
            y = random.uniform(0, HEIGHT)
            agents.append(SlimeAgent(x, y))
    elif mode == "few_sources":
        spawn_points = [(random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)) for _ in range(3)]
        for _ in range(AGENT_COUNT):
            sx, sy = random.choice(spawn_points)
            x = sx + random.uniform(-20, 20)
            y = sy + random.uniform(-20, 20)
            agents.append(SlimeAgent(x, y))
    return agents

agents = create_agents(SPAWN_MODE)

# --- Main loop ---
running = True
while running:
    screen.fill((0, 0, 0))
    click = False
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
            if event.button == 1:  # Left click = add food
                food_sources.append(mouse_pos)

    # Agent behavior
    for agent in agents:
        agent.sense()
        agent.move()

    # Update trails
    trail_map *= TRAIL_DECAY

    # Add food attractants
    for fx, fy in food_sources:
        pygame.draw.circle(screen, (100, 155, 0), (fx, fy), 3)
        if 0 <= fx < WIDTH and 0 <= fy < HEIGHT:
            trail_map[fx][fy] += 5

    # Draw trail map
    surface = pygame.surfarray.make_surface(np.rot90((np.clip(trail_map, 0, 255)).astype(np.uint8)))
    surface.set_alpha(255)
    screen.blit(surface, (0, 0))

    # Draw buttons
    clear_btn = pygame.Rect(10, 10, 100, 30)
    mode_btn = pygame.Rect(120, 10, 160, 30)
    action = draw_button(clear_btn, "Clear Trails", mouse_pos, click, "clear")
    mode_text = "Mode: " + ("Grid" if SPAWN_MODE == "grid" else "Few Sources")
    action2 = draw_button(mode_btn, mode_text, mouse_pos, click, "toggle_mode")

    if action == "clear":
        trail_map.fill(0)
        food_sources = []
    if action2 == "toggle_mode":
        SPAWN_MODE = "few_sources" if SPAWN_MODE == "grid" else "grid"
        agents = create_agents(SPAWN_MODE)
        trail_map.fill(0)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
