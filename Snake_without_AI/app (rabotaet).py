import pygame
import random

pygame.init()

WIDTH, HEIGHT = 980, 630
BLOCK_SIZE = 35
FPS = 15
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Змейка')
clock = pygame.time.Clock()

class Food(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

class SnakeSegment(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

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
        self.direction = (1 , 0)

    def draw(self):
        for i in self.segments:
            screen.blit(i.image, i.rect)


class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = self.generate_food()
        self.total_reward = 0

    def step(self, action):
        reward = 0
        if action == 0:
            self.snake.change_direction((0, -1))
        elif action == 1:
            self.snake.change_direction((0, 1))
        elif action == 2:
            self.snake.change_direction((-1, 0))
        elif action == 3:
            self.snake.change_direction((1, 0))

        self.snake.move()

        if pygame.sprite.collide_rect(self.snake.segments[-1], self.food):
            self.snake.segments.insert(0 , SnakeSegment(self.snake.segments[0].rect.x , self.snake.segments[0].rect.y))
            self.food = self.generate_food()
            reward = 100


        for i in self.snake.segments[:-1]:
            if pygame.sprite.collide_rect(i, self.snake.segments[-1]):
                self.snake.reset()
                reward = -100

        head = self.snake.segments[-1]
        if head.rect.x < 0 or head.rect.x >= WIDTH or head.rect.y < 0 or head.rect.y >= HEIGHT:
            self.snake.reset()
            reward = -100
        else:
            self.snake.segments.pop(0)

        return reward


    def generate_food(self):
        x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        return Food(x, y)


running = True
game = Game()

while running:
    screen.fill(GREEN)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action = None
    if keys[pygame.K_UP]:
        action = 0
    elif keys[pygame.K_DOWN]:
        action = 1
    elif keys[pygame.K_LEFT]:
        action = 2
    elif keys[pygame.K_RIGHT]:
        action = 3

    if action is not None:
        reward = game.step(action)

    game.snake.draw()
    screen.blit(game.food.image, game.food.rect)


    pygame.display.update()
    clock.tick(FPS)

pygame.quit()

