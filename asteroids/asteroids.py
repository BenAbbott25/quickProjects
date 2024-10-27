import pygame
import numpy as np
from vector import Vector

pygame.init()

class Game:
    def __init__(self, screen_width, screen_height, manual=False):
        self.running = True
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Asteroids")
        self.clock = pygame.time.Clock()
        self.asteroids = []
        self.player = Player(self)
        self.asteroid_spawn_timer = 0
        self.asteroid_spawn_interval = 1000
        self.max_asteroids = 1
        self.max_asteroids_counter = 0
        self.bullets = []
        self.manual = manual

    def spawn_asteroids(self):
        self.asteroid_spawn_timer += self.clock.get_time()
        print(f"Timer: {self.asteroid_spawn_timer} / {self.asteroid_spawn_interval}, Asteroids: {len(self.asteroids)} / {self.max_asteroids}, Counter: {self.max_asteroids_counter}")
        if self.asteroid_spawn_timer > self.asteroid_spawn_interval and len(self.asteroids) < self.max_asteroids:
            self.asteroid_spawn_timer = 0
            self.asteroid_spawn_interval = np.random.randint(100, int(np.floor(2000 / self.max_asteroids)))
            self.asteroids.append(Asteroid(self))

    def update(self, input):
        self.screen.fill((0, 0, 0))
        self.spawn_asteroids()
        for asteroid in self.asteroids:
            asteroid.move()
            asteroid.draw(self.screen)
            if asteroid.check_off_screen():
                self.asteroids.remove(asteroid)
                self.max_asteroids_counter += 1
                if self.max_asteroids_counter > (self.max_asteroids) ** 2:
                    self.max_asteroids += 1 if self.max_asteroids < 10 else 10
                    self.max_asteroids_counter = 0
        for bullet in self.bullets:
            bullet.update()

        self.player.update(input)
        pygame.display.flip()

class Player:
    def __init__(self, game):
        self.game = game
        self.x = self.game.screen_width / 2
        self.y = self.game.screen_height / 2
        self.thrust_vector = Vector(0, np.pi / 2, polar=True)
        self.velocity_vector = Vector(0, 0)
        self.max_health = 100
        self.health = self.max_health
        self.size = 10

    def move(self):
        self.x += self.velocity_vector.x
        self.y += self.velocity_vector.y

        self.velocity_vector.x += self.thrust_vector.x
        self.velocity_vector.y += self.thrust_vector.y
        self.velocity_vector.update_polar()
        self.velocity_vector.magnitude *= 0.9
        if self.velocity_vector.magnitude > 10:
            self.velocity_vector.magnitude = 10
        if self.velocity_vector.magnitude < 0.01:
            self.velocity_vector.magnitude = 0
        self.velocity_vector.update_cartesian()

    def draw(self, screen):
        self.draw_flame(screen)
        front_point = (self.x + self.size * np.cos(self.thrust_vector.angle), self.y + self.size * np.sin(self.thrust_vector.angle))
        back_left_point = (self.x - self.size * np.sin(self.thrust_vector.angle + np.pi / 4), self.y + self.size * np.cos(self.thrust_vector.angle + np.pi / 4))
        back_right_point = (self.x + self.size * np.sin(self.thrust_vector.angle - np.pi / 4), self.y - self.size * np.cos(self.thrust_vector.angle - np.pi / 4))
        pygame.draw.line(screen, (155, 255, 155), back_left_point, front_point, 2)
        pygame.draw.line(screen, (155, 255, 155), back_left_point, (self.x, self.y), 2)
        pygame.draw.line(screen, (155, 255, 155), (self.x, self.y), back_right_point, 2)
        pygame.draw.line(screen, (155, 255, 155), back_right_point, front_point, 2)
        self.draw_health(screen)

    def draw_flame(self, screen):
        flame_point = (self.x - 10 * self.thrust_vector.magnitude * np.cos(self.thrust_vector.angle), self.y - 10 * self.thrust_vector.magnitude * np.sin(self.thrust_vector.angle))
        flame_source_left = (self.x - 4 * np.cos(self.thrust_vector.angle + np.pi / 4), self.y - 4 * np.sin(self.thrust_vector.angle + np.pi / 4))
        flame_source_right = (self.x - 4 * np.cos(self.thrust_vector.angle - np.pi / 4), self.y - 4 * np.sin(self.thrust_vector.angle - np.pi / 4))
        pygame.draw.line(screen, (255, 255, 155), flame_source_left, flame_point, int(np.floor(self.thrust_vector.magnitude * 3)))
        pygame.draw.line(screen, (255, 255, 155), flame_source_right, flame_point, int(np.floor(self.thrust_vector.magnitude * 3)))
        # pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (self.game.screen_width / 2, self.game.screen_height / 2), 2)
        # pygame.draw.line(screen, (255, 255, 255), (self.game.screen_width / 2, self.game.screen_height / 2), (self.game.screen_width / 2 + 10 * np.cos(self.thrust_vector.angle), self.game.screen_height / 2 + 10 * np.sin(self.thrust_vector.angle)), 2)

    def draw_health(self, screen):
        health_percentage = self.health / self.max_health
        colour = (int(np.floor(255 - health_percentage * 255)), int(np.floor(health_percentage * 255)), 0)
        pygame.draw.rect(screen, colour, (0, 0, health_percentage * self.game.screen_width, 10))

    def check_collision(self, asteroids):
        for asteroid in asteroids:
            if self.x - self.size < asteroid.x + asteroid.size and self.x + self.size > asteroid.x - asteroid.size and self.y - self.size < asteroid.y + asteroid.size and self.y + self.size > asteroid.y - asteroid.size:
                self.health -= asteroid.size
    
    def handle_input(self, input):
        if self.game.manual:
            if input['left']:
                self.thrust_vector.angle -= 0.1
                if self.thrust_vector.angle < -np.pi:
                    self.thrust_vector.angle += 2 * np.pi
            if input['right']:
                self.thrust_vector.angle += 0.1
                if self.thrust_vector.angle > np.pi:
                    self.thrust_vector.angle -= 2 * np.pi
            if input['up']:
                self.thrust_vector.magnitude += 0.1 if self.thrust_vector.magnitude < 1 else 0
            if input['down']:
                self.thrust_vector.magnitude -= 0.1 if self.thrust_vector.magnitude > 0 else 0

        else:
            self.thrust_vector.magnitude += input[0]
            self.thrust_vector.angle += input[1]

            if self.thrust_vector.magnitude > 1:
                self.thrust_vector.magnitude = 1
            elif self.thrust_vector.magnitude < 0:
                self.thrust_vector.magnitude = 0

            self.thrust_vector.angle = (self.thrust_vector.angle + np.pi) % (2 * np.pi) - np.pi

        self.thrust_vector.update_cartesian()

        if self.x < 50 or self.x > self.game.screen_width - 50 or self.y < 50 or self.y > self.game.screen_height - 50:
            angle_to_centre = np.arctan2(self.y - self.game.screen_height / 2, self.x - self.game.screen_width / 2)
            angle_diff = angle_to_centre - self.thrust_vector.angle
            
            # Normalize angle_diff to be within [-π, π]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            
            # print(f'thrust vector angle: {self.thrust_vector.angle}, Centre angle: {angle_to_centre}, Angle diff: {angle_diff}')
            
            # Adjust the thrust vector angle
            self.thrust_vector.angle -= angle_diff * 0.1
            
            # Normalize thrust vector angle to be within [-π, π]
            self.thrust_vector.angle = (self.thrust_vector.angle + np.pi) % (2 * np.pi) - np.pi
            
            # Update the thrust vector's cartesian coordinates
            self.thrust_vector.update_cartesian()

        if input['space']:
            self.shoot()

    def update(self, input):
        self.handle_input(input)
        self.draw(self.game.screen)
        self.move()
        self.check_collision(self.game.asteroids)
        if self.health <= 0:
            print("Game Over")
            self.game.running = False

    def shoot(self):
        if len(self.game.bullets) < 1:
            self.game.bullets.append(Bullet(self))

class Bullet:
    def __init__(self, player):
        self.player = player
        self.x = player.x
        self.y = player.y
        self.velocity_vector = Vector(25, player.thrust_vector.angle, polar=True)
        self.velocity_vector.update_cartesian()

    def move(self):
        self.x += self.velocity_vector.x
        self.y += self.velocity_vector.y

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), 1)

    def check_collision(self, asteroids):
        for asteroid in asteroids:
            if self.x - 1 < asteroid.x + asteroid.size and self.x + 1 > asteroid.x - asteroid.size and self.y - 1 < asteroid.y + asteroid.size and self.y + 1 > asteroid.y - asteroid.size:
                self.player.game.asteroids.remove(asteroid)
                self.player.game.max_asteroids_counter += 1
                self.player.game.bullets.remove(self)

    def check_off_screen(self):
        if self.x < -100 or self.x > self.player.game.screen_width + 100 or self.y < -100 or self.y > self.player.game.screen_height + 100:
            self.player.game.bullets.remove(self)
            return True
        return False
    
    def update(self):
        self.move()
        self.draw(self.player.game.screen)
        self.check_collision(self.player.game.asteroids)
        self.check_off_screen()

class Asteroid:
    def __init__(self, game):
        self.game = game
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            self.x = np.random.randint(0, self.game.screen_width)
            self.y = -50
        elif edge == 'bottom':
            self.x = np.random.randint(0, self.game.screen_width)
            self.y = self.game.screen_height + 50
        elif edge == 'left':
            self.x = -50
            self.y = np.random.randint(0, self.game.screen_height)
        else:  # right
            self.x = self.game.screen_width + 50
            self.y = np.random.randint(0, self.game.screen_height)
        
        target_x = np.random.randint(50, self.game.screen_width - 50)
        target_y = np.random.randint(50, self.game.screen_height - 50)
        direction_x = (target_x - self.x) / 100
        direction_y = (target_y - self.y) / 100
        self.velocity_vector = Vector(direction_x, direction_y)
        self.velocity_vector.update_polar()
        self.size = np.random.randint(5, 15)

    def move(self):
        self.x += self.velocity_vector.x
        self.y += self.velocity_vector.y

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), self.size)

    def check_off_screen(self):
        if self.x < -75 or self.x > self.game.screen_width + 75 or self.y < -75 or self.y > self.game.screen_height + 75:
            return True
        return False

def main(manual=True):
    game = Game(800, 800, manual)
    while game.running:
        if game.manual:
            input = pygame.key.get_pressed()
            input = {
            'left': input[pygame.K_LEFT],
            'right': input[pygame.K_RIGHT],
            'up': input[pygame.K_UP],
            'down': input[pygame.K_DOWN],
            'space': input[pygame.K_SPACE]
            }
        game.update(input)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
        game.clock.tick(60)

if __name__ == "__main__":
    main(manual=True)