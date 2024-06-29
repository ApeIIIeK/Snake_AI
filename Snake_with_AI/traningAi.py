import pygame
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import threading



pygame.init()
WIDTH, HEIGHT = 980, 630
BLOCK_SIZE = 35
FPS = 15
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
GREY = (41, 49, 51)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Змейка')
clock = pygame.time.Clock()
FONT = pygame.font.Font(None, 36)

# Класс еды
class Food(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
# класс одного сегмента змеи для создания полноценной змеи
class SnakeSegment(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
# Класс змеи
class Snake:
    def __init__(self):
        self.segments = [SnakeSegment(BLOCK_SIZE * i, BLOCK_SIZE * 5) for i in range(5, 8)]
        self.direction = (1, 0) # x,y

    def move(self):
        head = self.segments[-1]
        x, y = head.rect.x, head.rect.y
        self.segments.append(SnakeSegment(x + BLOCK_SIZE * self.direction[0], y + BLOCK_SIZE * self.direction[1]))
        head = self.segments[-1]

    def change_direction(self, new):
        if self.direction != (new[0] * -1, new[1] * -1):
            self.direction = new

    def reset(self):
        self.segments = [SnakeSegment(BLOCK_SIZE * i , BLOCK_SIZE * 5) for i in range(5 , 8)]
        self.direction = (1, 0)

    def draw(self):
        for i in self.segments:
            screen.blit(i.image, i.rect)
# Класс игры
class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = self.generate_food()
        self.attempt = 1
        self.total_reward = 0
        self.count_move_to_death = 0
        self.score = 0
        self.record_score = 0

    # Выполняет шаг игры, основываясь на действии агента столкновение с едой и стеной, возвращяет награду
    def step(self, action):
        reward = 0
        done = False
        if action == 0:
            self.snake.change_direction((0, -1))
        elif action == 1:
            self.snake.change_direction((0, 1))
        elif action == 2:
            self.snake.change_direction((-1, 0))
        elif action == 3:
            self.snake.change_direction((1, 0))

        old_head = self.snake.segments[-1]

        self.snake.move()
        self.count_move_to_death += 1

        if pygame.sprite.collide_rect(self.snake.segments[-1], self.food):
            #self.snake.segments.insert(0 , SnakeSegment(self.snake.segments[0].rect.x , self.snake.segments[0].rect.y))
            self.food = self.generate_food()
            reward = 100
            self.score += 1


        head = self.snake.segments[-1]
        if head.rect.x < 0 or head.rect.x >= WIDTH or head.rect.y < 0 or head.rect.y >= HEIGHT:
            done = True
            self.snake.reset()
            reward = -100
            self.attempt += 1
            self.count_move_to_death = 0
            self.score = 0
        else:
            self.snake.segments.pop(0)

        distance_old = abs(old_head.rect.x - self.food.rect.x) + abs(old_head.rect.y - self.food.rect.y)
        distance_new = abs(head.rect.x - self.food.rect.x) + abs(head.rect.y - self.food.rect.y)
        if distance_old > distance_new:
            reward += 1
        else:
            reward -= 1

        if self.record_score < self.score:
            self.record_score = self.score
            update_target_model(model, target_model)
            model.save(f"models/snake_ai_model_{self.attempt}_new_record.keras")
        if self.count_move_to_death > 40:
            self.snake.reset()
            self.food = self.generate_food()
            self.attempt += 1
            self.count_move_to_death = 0
            self.score = 0


        return get_game_state(self.snake, self.food,WIDTH,HEIGHT), reward, done



    # Функция генерации еды в рандомном месте
    def generate_food(self):
        x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        return Food(x, y)

# Создание модели ии
def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(24, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model
# тренировачная модель
def train_model(memory, model, target_model, gamma, batch_size):
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * np.amax(target_model.predict(next_state)[0])
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)

def update_target_model(model, target_model):
    target_model.set_weights(model.get_weights())
# информация передаваемая ии
def get_game_state(snake, food, width, height):
    state = []
    head_x, head_y = snake.segments[-1].rect.x, snake.segments[-1].rect.y
    x_change, y_change = snake.direction

    danger_straight = (
            (x_change == 1 and head_x + BLOCK_SIZE >= width) or
            (x_change == -1 and head_x - BLOCK_SIZE < 0) or
            (x_change == 0 and y_change == -1 and head_y - BLOCK_SIZE < 0) or
            (x_change == 0 and y_change == 1 and head_y + BLOCK_SIZE >= height)
    )

    danger_right = (
            (x_change == 0 and y_change == -1 and head_x + BLOCK_SIZE >= width) or
            (x_change == 0 and y_change == 1 and head_x - BLOCK_SIZE < 0) or
            (x_change == -1 and head_y - BLOCK_SIZE < 0) or
            (x_change == 1 and head_y + BLOCK_SIZE >= height)
    )

    danger_left = (
            (x_change == 0 and y_change == 1 and head_x + BLOCK_SIZE >= width) or
            (x_change == 0 and y_change == -1 and head_x - BLOCK_SIZE < 0) or
            (x_change == 1 and head_y - BLOCK_SIZE < 0) or
            (x_change == -1 and head_y + BLOCK_SIZE >= height)
    )

    state.extend([
        danger_straight,
        danger_right,
        danger_left,
        x_change == -1,
        x_change == 1,
        y_change == 1,
        y_change == -1,
        food.rect.x < head_x,
        food.rect.x > head_x,
        food.rect.y < head_y,
        food.rect.y > head_y
    ])

    state = [1 if s else 0 for s in state]
    return np.array(state)
# функция для отрисовки текста
def draw_text(surface, text, position):
    # Рисует текст на экране
    font = FONT
    img = font.render(text, True, GREY)
    return surface.blit(img, position)

running = True
game = Game()

input_shape = len(get_game_state(game.snake,game.food,WIDTH,HEIGHT))
output_shape = 4
model = build_model(input_shape, output_shape)
target_model = build_model(input_shape, output_shape)
update_target_model(model, target_model)

memory = deque(maxlen=8000)
batch_size = 800
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
reward = None
next_state = None
training_thread = None

# Игровой цикл
while running:
    screen.fill(GREEN)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            model.save(f"models/snake_ai_model_final.keras")


    state = get_game_state(game.snake, game.food, WIDTH, HEIGHT)
    next_state_size = len(state)
    state = np.reshape(state, [1, next_state_size])
    if np.random.rand() <= epsilon:
        action = random.randint(0, 3)
    else:
        action_value = model.predict(state)
        action = np.argmax(action_value[0])



    next_state, reward, done = game.step(action)
    next_state_size = len(next_state)
    next_state = np.reshape(state, [1, next_state_size])

    memory.append((state, action, reward, next_state, done))

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if len(memory) > batch_size and (training_thread is None or not training_thread.is_alive()):
        training_thread = threading.Thread(target=train_model, args=(memory, model, target_model, gamma, batch_size))
        training_thread.start()

    print(str(game.score) + '>' + str(game.record_score))
    if done:
        if game.score > 40:
            print("Final")
            model.save(f"models/snake_ai_model_final.keras")
            break
        else:
            game.food = game.generate_food()



    game.snake.draw()
    screen.blit(game.food.image, game.food.rect)

    draw_text(screen, f'Attempt: {game.attempt}', (35, 100))
    draw_text(screen, f'count_step_to_death: {game.count_move_to_death}', (35, 135))
    draw_text(screen, f'score: {game.score}', (35, 170))
    draw_text(screen, f'record_score: {game.record_score}', (35, 205))

    pygame.display.update()
    clock.tick(FPS)




pygame.quit()

