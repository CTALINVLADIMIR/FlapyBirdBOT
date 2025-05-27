import pygame
import sys
import random
import numpy as np
from collections import defaultdict
import pickle
import os

# Инициализация PyGame
pygame.init()

# Настройки экрана
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird с ИИ")

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Параметры игры
BIRD_SIZE = 30
GRAVITY = 0.5
JUMP_STRENGTH = -8
WALL_WIDTH = 60
WALL_GAP = 200
WALL_SPEED = 3

# Параметры ИИ
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 1000000
SHOW_EVERY = 100
epsilon = 0.2
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01

# Дискретизация пространства состояний
DISCRETE_BINS = 20
BUCKETS = {
    'bird_y': np.linspace(0, HEIGHT, DISCRETE_BINS),
    'bird_vel': np.linspace(-15, 15, DISCRETE_BINS),
    'next_wall_x': np.linspace(0, WIDTH, DISCRETE_BINS),
    'next_wall_top': np.linspace(0, HEIGHT - WALL_GAP, DISCRETE_BINS)
}

class Bird:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = WIDTH // 4
        self.y = HEIGHT // 2
        self.vel = 0
        self.alive = True
        self.score = 0
        self.total_reward = 0
        self.rect = pygame.Rect(self.x, self.y, BIRD_SIZE, BIRD_SIZE)
    
    def reset_near_gap(self, gap_y):
        self.y = gap_y + WALL_GAP // 2
        self.vel = 0
        self.alive = True
        self.rect = pygame.Rect(self.x, self.y, BIRD_SIZE, BIRD_SIZE)
    
    def jump(self):
        self.vel = JUMP_STRENGTH
    
    def update(self):
        self.vel += GRAVITY
        self.y += self.vel
        self.rect.y = self.y
        
        # Проверка границ экрана (смерть при касании)
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.alive = False

class Wall:
    def __init__(self, x=WIDTH):
        self.gap_y = random.randint(50, HEIGHT - 50 - WALL_GAP)
        self.x = x
        self.passed = False
        self.top_rect = pygame.Rect(self.x, 0, WALL_WIDTH, self.gap_y)
        self.bottom_rect = pygame.Rect(self.x, self.gap_y + WALL_GAP, WALL_WIDTH, HEIGHT - self.gap_y - WALL_GAP)
    
    def update(self):
        self.x -= WALL_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x
    
    def collides(self, bird_rect):
        return bird_rect.colliderect(self.top_rect) or bird_rect.colliderect(self.bottom_rect)
    
    def is_passed(self, bird_rect):
        return not self.passed and bird_rect.left > self.x + WALL_WIDTH
    
    def is_off_screen(self):
        return self.x < -WALL_WIDTH

def get_discrete_state(bird, walls):
    if not walls:
        wall = Wall(WIDTH)
    else:
        wall = walls[0]
    
    state = (
        bird.y,
        bird.vel,
        wall.x,
        wall.gap_y
    )
    
    discrete_state = []
    for i, key in enumerate(['bird_y', 'bird_vel', 'next_wall_x', 'next_wall_top']):
        discrete_state.append(np.digitize(state[i], BUCKETS[key]) - 1)
    
    return tuple(discrete_state)

# Q-таблица
Q_FILE = "qtable.pkl"
if os.path.exists(Q_FILE):
    with open(Q_FILE, 'rb') as f:
        q_table = pickle.load(f)
    if isinstance(q_table, dict):
        q_table = defaultdict(lambda: np.array([0.0, 0.0]), q_table)
else:
    q_table = defaultdict(lambda: np.array([0.0, 0.0]))

def draw_game(bird, walls, episode, epsilon):
    screen.fill(BLACK)
    
    # Стены
    for wall in walls:
        pygame.draw.rect(screen, GREEN, wall.top_rect)
        pygame.draw.rect(screen, GREEN, wall.bottom_rect)
    
    # Птица
    color = RED if not bird.alive else BLUE
    pygame.draw.rect(screen, color, bird.rect)
    
    # Информация
    font = pygame.font.SysFont('Arial', 24)
    texts = [
        f"Эпизод: {episode}",
        f"Очки: {bird.score}",
        f"Награда: {int(bird.total_reward)}",
        f"Epsilon: {epsilon:.3f}"
    ]
    
    for i, text in enumerate(texts):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, 10 + i * 30))
    
    pygame.display.flip()

def save_qtable():
    with open(Q_FILE, 'wb') as f:
        pickle.dump(dict(q_table), f)

# Основной цикл обучения
clock = pygame.time.Clock()

for episode in range(EPISODES):
    show_game = episode % SHOW_EVERY == 0
    bird = Bird()
    walls = [Wall()]
    
    running = True
    while running:
        if show_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_qtable()
                    pygame.quit()
                    sys.exit()
        
        if not bird.alive:
            bird.reset_near_gap(walls[0].gap_y)
            continue
        
        # Получение состояния
        discrete_state = get_discrete_state(bird, walls)
        
        # Выбор действия
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, 2)
        
        # Действие
        if action == 1:
            bird.jump()
        
        # Обновление
        bird.update()
        for wall in walls:
            wall.update()
        
        # Проверка столкновений
        reward = 0.1
        
        # Смерть при касании стен или границ экрана
        if walls[0].collides(bird.rect) or not bird.alive:
            reward = -100
            bird.alive = False
        elif walls[0].is_passed(bird.rect):
            walls[0].passed = True
            bird.score += 1
            reward = 50
        
        # Добавление новых стен
        if walls[-1].x < WIDTH - 400:
            walls.append(Wall())
        
        # Удаление стен за экраном
        if walls[0].is_off_screen():
            walls.pop(0)
        
        # Новое состояние
        new_discrete_state = get_discrete_state(bird, walls)
        
        # Обновление Q-таблицы
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state][action]
        
        if not bird.alive:
            new_q = reward
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[discrete_state][action] = new_q
        bird.total_reward += reward
        
        # Отрисовка
        if show_game:
            draw_game(bird, walls, episode, epsilon)
            clock.tick(60)
        
        # Завершение эпизода
        if bird.score >= 50 or len(walls) > 10:
            running = False
    
    # Обновление epsilon
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    
    # Сохранение Q-таблицы
    if episode % 100 == 0:
        save_qtable()
    
    print(f"Эпизод {episode}: Очки {bird.score}, Награда {int(bird.total_reward)}, Epsilon {epsilon:.3f}")

save_qtable()
pygame.quit()
sys.exit()