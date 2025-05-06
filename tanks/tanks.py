import pygame
import numpy as np

class Tank:
    def __init__(self, game, id, position, color, player=True):
        self.game = game
        self.id = id
        self.position = position
        self.color = color
        self.size = 10
        self.body_angle = 0.0
        self.turret_angle = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.max_speed = 5
        self.bullet_speed = 10
        self.bullet_size = 2
        self.bullet_color = (255, 255, 255)
        self.bullets = []
        self.reload_time = 50
        self.reload_timer = 0
        self.max_health = 100
        self.health = self.max_health
        self.player = player
        self.fitness = 0

    def shoot(self):
        if len(self.bullets) < 10 and self.reload_timer == 0:
            bullet = Bullet(self, self.position, self.turret_angle, self.bullet_speed, self.bullet_size, self.bullet_color)
            self.bullets.append(bullet)
            self.reload_timer = self.reload_time
        elif self.reload_timer > 0:
            self.reload_timer -= 1

    def draw(self, screen):
        points = [
            (self.position[0] + self.size * np.cos(self.body_angle + np.pi / 6), self.position[1] + self.size * np.sin(self.body_angle + np.pi / 6)),
            (self.position[0] + self.size * np.cos(self.body_angle - np.pi / 6), self.position[1] + self.size * np.sin(self.body_angle - np.pi / 6)),
            (self.position[0] - self.size * np.cos(self.body_angle + np.pi / 6), self.position[1] - self.size * np.sin(self.body_angle + np.pi / 6)),
            (self.position[0] - self.size * np.cos(self.body_angle - np.pi / 6), self.position[1] - self.size * np.sin(self.body_angle - np.pi / 6)),
        ]
        pygame.draw.polygon(screen, self.color, points)

        turret_points = [
            (self.position[0] + self.size * 1.5 * np.cos(self.turret_angle + np.pi / 30), self.position[1] + self.size * 1.5 * np.sin(self.turret_angle + np.pi / 30)),
            (self.position[0] + self.size * 1.5 * np.cos(self.turret_angle - np.pi / 30), self.position[1] + self.size * 1.5 * np.sin(self.turret_angle - np.pi / 30)),
            (self.position[0] - self.size * 0.5 * np.cos(self.turret_angle + np.pi / 30), self.position[1] - self.size * 0.5 * np.sin(self.turret_angle + np.pi / 30)),
            (self.position[0] - self.size * 0.5 * np.cos(self.turret_angle - np.pi / 30), self.position[1] - self.size * 0.5 * np.sin(self.turret_angle - np.pi / 30)),
        ]
        pygame.draw.polygon(screen, self.color, turret_points)

        # draw health bar
        health_percentage = self.health / self.max_health
        health_bar_colour = (255 * (1 - health_percentage), 255 * health_percentage, 0)

        start_angle = 0
        end_angle = 2 * np.pi * health_percentage

        pygame.draw.arc(
            screen, 
            health_bar_colour, 
            (self.position[0] - self.size * 2, self.position[1] - self.size * 2, self.size * 4, self.size * 4), 
            start_angle, 
            end_angle, 
            2
        )

        for bullet in self.bullets:
            bullet.draw(screen)

    def update(self, control_inputs):
        if self.health <= 0:
            self.fitness -= 10
            self.game.fitnesses[self.id] = self.fitness
            self.game.inactivity_timer = 0
            del self.game.tanks[self.id]
            return
        
        # Control the player tank based on the genome outputs
        if control_inputs[0] > 0.5:
            self.body_angle -= np.pi / 100
        if control_inputs[1] > 0.5:
            self.body_angle += np.pi / 100
        if control_inputs[2] > 0.5:
            self.velocity += np.array([np.cos(self.body_angle), np.sin(self.body_angle)]) * self.max_speed
        if control_inputs[3] > 0.5:
            self.velocity -= np.array([np.cos(self.body_angle), np.sin(self.body_angle)]) * self.max_speed
        if control_inputs[4] > 0.5:
            self.turret_angle -= np.pi / 100
        if control_inputs[5] > 0.5:
            self.turret_angle += np.pi / 100
        if control_inputs[6] > 0.5:
            self.shoot()

        # Update position and velocity
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed
        self.position += self.velocity / 10

        # Prevent tanks from leaving the screen
        self.position[0] = max(0, min(self.position[0], 800))
        self.position[1] = max(0, min(self.position[1], 600))

        if np.linalg.norm(self.velocity) > 0.05:
            self.velocity *= 0.99
            self.velocity = np.array([np.cos(self.body_angle), np.sin(self.body_angle)]) * np.linalg.norm(self.velocity)
        else:
            self.velocity = np.array([0.0, 0.0])

        for bullet in self.bullets:
            bullet.update()

class Bullet:
    def __init__(self, tank, position, angle, speed, size, color):
        self.tank = tank
        self.position = np.copy(position) + np.array([np.cos(angle), np.sin(angle)]) * 10
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        self.size = size
        self.color = color

    def update(self):
        self.position += self.velocity
        # Check if the bullet is out of bounds
        if self.position[0] < 0 or self.position[0] > screen_width or self.position[1] < 0 or self.position[1] > screen_height:
            if self in self.tank.bullets:  # Ensure the bullet is in the list before removing
                self.tank.bullets.remove(self)
            return

        # Check for collision with tanks
        for tank_id in self.tank.game.tanks:
            tank = self.tank.game.tanks[tank_id]
            if tank != self.tank:
                if np.linalg.norm(self.position - tank.position) < self.size + tank.size:
                    if self in self.tank.bullets:  # Ensure the bullet is in the list before removing
                        self.tank.bullets.remove(self)
                    tank.health -= 25
                    self.tank.fitness += 1
                    tank.fitness -= 1
                    if tank.health <= 0:
                        self.tank.fitness += 10
                    return

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.position, self.size)

class Game:
    def __init__(self, watch_game=False):
        pygame.init()
        global screen_width, screen_height
        screen_width = 800
        screen_height = 600
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.tanks = {}
        self.inactivity_timer = 0
        self.max_inactivity_timer = 500
        self.running = True
        self.fitnesses = {}
        self.watch_game = watch_game
        
    def update(self):
        self.inactivity_timer += 1
        if self.inactivity_timer > self.max_inactivity_timer or len(self.tanks) == 1:
            self.running = False
            self.fitnesses.update({tank_id: tank.fitness for tank_id, tank in self.tanks.items()})
        
        if self.watch_game:
            self.screen.fill((0, 0, 0))
            for tank in self.tanks:
                self.tanks[tank].draw(self.screen)
            

            # draw inactivity timer rect
            inactivity_percentage = self.inactivity_timer / self.max_inactivity_timer
            bar_colour = (255 * inactivity_percentage, 255 * (1 - inactivity_percentage), 0)
            pygame.draw.rect(self.screen, bar_colour, (0, 0, screen_width * (1 - inactivity_percentage), screen_height * 0.01))
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    game = Game(num_tanks=10)  # Specify the number of tanks
    game.run()