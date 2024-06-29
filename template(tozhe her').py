import pygame
import random

pygame.init()

WIDTH, HEIGHT = 980, 630
BLOCK_SIZE = 35
FPS = 10
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
FONT = pygame.font.Font(None, 36)

class Food(pygame.sprite.Sprite):
    # Создает объект еды
    def __init__(self, x, y):
        pass

class SnakeSegment(pygame.sprite.Sprite):
    # Создает сегмент змейки
    def __init__(self, x, y):
        pass

class Snake:
    # Инициализирует змейку
    def __init__(self):
        pass

    # Двигает змейку в текущем направлении
    def move(self):
        pass

    # Сбрасывает змейку в начальное состояние
    def reset(self):
        pass

    # Меняет направление движения змейки
    def change_direction(self, direction):
        pass

    # Рисует змейку на поверхности
    def draw(self, surface):
        pass

class Game:
    # Инициализирует игру
    def __init__(self):
        pass

    # Генерирует новую еду в случайном месте
    def generate_food(self):
        pass

    # Выполняет шаг игры, основываясь на действии агента столкновение с едой (добавляем сегмент)
    # столконовение со стенами
    # ВОЗВРАЩАЕТ НАГРАДУ
    # -100 - стене
    # +10 еду
    def step(self, action):
        pass

    # Возвращает текущее состояние игры
    def get_state(self):
        pass

def get_game_state(snake, food, width, height):
    # Возвращает состояние игры для агента
    pass

def draw_text(surface, text, position):
    # Рисует текст на экране
    pass

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Змейка')

    game = Game()

    clock = pygame.time.Clock()
    running = True
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
            state, reward = game.step(action)

        game.snake.draw(screen)
        screen.blit(game.food.image, game.food.rect)

        draw_text(screen, f'Total Reward: {game.total_reward}', (10, 10))
        draw_text(screen, f'Attempt: {game.attempt}', (10, 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    # Запускает основную функцию, если скрипт выполнен напрямую
    main()
