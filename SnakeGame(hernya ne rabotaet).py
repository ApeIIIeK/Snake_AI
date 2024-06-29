import pygame
import random

pygame.init()

WIDTH, HEIGHT = 980, 630
BLOCK_SIZE = 35
FPS = 10
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Змейка')
clock = pygame.time.Clock()

class SnakeSegment(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(topleft=(x, y))

class Snake:
    def __init__(self,food):
        self.food = food
        self.segments = pygame.sprite.Group()
        self.create_snake()
        self.direction = pygame.math.Vector2(BLOCK_SIZE, 0)

    def move(self):
        segments = list(self.segments)[::-1]
        if not segments:  # Проверяем, не пуст ли список
            return  
        # Проверяем столкновение с самим собой
        head_segment = segments[-1]
        for segment in segments[:-1]:
            if head_segment.rect.colliderect(segment.rect):
                self.reset()
        
        # Перемещаем каждый сегмент на место следующего
        for i in range(len(segments) - 1):
            segments[i].rect.topleft = segments[i + 1].rect.topleft
        # Перемещаем голову змейки в заданном направлении
        head_segment.rect.move_ip(self.direction)
        if (head_segment.rect.left < 0 or head_segment.rect.right > WIDTH or
            head_segment.rect.top < 0 or head_segment.rect.bottom > HEIGHT):
            self.reset()
        head_segment = list(self.segments.sprites())[-1]
        self.eat_food()
        

    def respawn_food(self):
        pass
        
    def grow(self,):
        tail_segment = list(self.segments.sprites())[0]
        new_segment = SnakeSegment(tail_segment.rect.x, tail_segment.rect.y)
        self.segments.add(new_segment)
        
    
        
    def change_direction(self, new_direction):
        # Проверяем, не противоположно ли новое направление текущему
        if new_direction == (-self.direction[0], -self.direction[1]):
            return
        # Изменяем направление движения змейки
        self.direction = new_direction

    def reset(self):
        self.segments.empty()  # Удаляем все сегменты змейки
        self.create_snake()  # Создаем змейку заново
        self.direction = pygame.math.Vector2(BLOCK_SIZE, 0)  # Сбрасываем направление движения

    def create_snake(self):
        for i in range(12):
            x = WIDTH // 2 - BLOCK_SIZE * i
            y = HEIGHT // 2
            segment = SnakeSegment(x, y)
            self.segments.add(segment)

    
    def draw(self):
        self.segments.draw(screen)

class Food(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect(topleft=(random.randrange(0, WIDTH, BLOCK_SIZE),
                                                 random.randrange(0, HEIGHT, BLOCK_SIZE)))

    def draw(self):
        screen.blit(self.image, self.rect)
        
class Game:
    def __init__(self):
        food = Food()
        self.snake = Snake(food)
        self.food = Food()

    def run(self):
        self.snake.draw()
        self.food.draw()
        self.snake.eat_food()
        
    def eat_food(self):
        head_segment = self.segments.sprites()[-1]
        if pygame.sprite.collide_rect(head_segment, food):
            print("Удар")
            self.grow()
            self.food = Food()
        
game = Game()
running = True

while running:
    screen.fill(GREEN)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_UP]:
            game.snake.change_direction((0, -BLOCK_SIZE))
            action = 0
        elif keys[pygame.K_DOWN]:
            game.snake.change_direction((0, BLOCK_SIZE))
            action = 1
        elif keys[pygame.K_LEFT]:
            game.snake.change_direction((-BLOCK_SIZE, 0))
            action = 2
        elif keys[pygame.K_RIGHT]:
            game.snake.change_direction((BLOCK_SIZE, 0))
            action = 3
       


    game.snake.move()
    game.run()

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()
