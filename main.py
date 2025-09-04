import math
import random
from collections import deque

import pygame

# -------------------------
# Configuration
# -------------------------
WIDTH, HEIGHT = 2880, 1620
ASPECT = WIDTH / HEIGHT
FPS = 60

PADDLE_W, PADDLE_H = 14, 128
PADDLE_MARGIN = 36
PADDLE_MAX_SPEED = 520  # px/s

BALL_SIZE = 14
BALL_SPEED = 420  # px/s initial
BALL_SPEED_MAX = 760
BALL_ACCEL_HIT = 1.04  # multiply on paddle hit
BALL_ACCEL_TIME = 0.000  # additive accel per frame (0 to disable)

WIN_SCORE = 7

GLOW_STRENGTH = 4  # 1..6
TRAIL_LENGTH = 30
PARTICLE_COUNT_HIT = 22
PARTICLE_COUNT_SCORE = 34

# Colors (neon palette)
C_BG = (10, 12, 16)
C_LINES = (18, 22, 28)
C_WHITE = (240, 248, 255)
C_CYAN = (0, 255, 255)
C_MAGENTA = (255, 0, 168)
C_YELLOW = (255, 220, 70)
C_GREEN = (0, 255, 140)
C_RED = (255, 60, 60)

# -------------------------
# Helpers
# -------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

# -------------------------
# Visual FX
# -------------------------
class Particle:
    def __init__(self, pos, vel, life, size, color):
        self.x, self.y = pos
        self.vx, self.vy = vel
        self.life = life
        self.size = size
        self.color = color
        self.age = 0.0

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 600 * dt * 0.25  # slight gravity for style
        self.age += dt
        return self.age < self.life

    def draw(self, surf):
        t = 1.0 - (self.age / self.life)
        a = max(0, min(255, int(255 * t)))
        r = max(1, int(self.size * (0.5 + 0.5 * t)))
        col = (*self.color[:3], a)
        pygame.draw.circle(surf, col, (int(self.x), int(self.y)), r)

class FX:
    def __init__(self):
        self.particles = []

    def burst(self, pos, count, base_color):
        for _ in range(count):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(80, 540)
            vel = (math.cos(ang) * spd, math.sin(ang) * spd)
            life = random.uniform(0.25, 0.85)
            size = random.randint(2, 4)
            color = base_color
            self.particles.append(Particle(pos, vel, life, size, color))

    def update(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)]

    def draw(self, glow_layer):
        for p in self.particles:
            p.draw(glow_layer)

def make_scanlines(width, height, spacing=3, alpha=24):
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    for y in range(0, height, spacing):
        pygame.draw.line(surf, (255, 255, 255, alpha), (0, y), (width, y))
    return surf

def make_vignette(width, height, strength=220):
    # radial gradient vignette
    vignette = pygame.Surface((width, height), pygame.SRCALPHA)
    cx, cy = width / 2, height / 2
    maxd = math.hypot(cx, cy)
    for y in range(height):
        for x in range(width):
            d = math.hypot(x - cx, y - cy)
            t = clamp((d - 0.6 * maxd) / (0.4 * maxd), 0.0, 1.0)
            a = int(t * strength)
            vignette.set_at((x, y), (0, 0, 0, a))
    return vignette

def draw_center_dashed_line(surf, color=(200, 200, 200), dash_len=16, gap=16, width=4):
    y = 0
    while y < HEIGHT:
        pygame.draw.rect(surf, color, (WIDTH // 2 - width // 2, y, width, dash_len))
        y += dash_len + gap

# -------------------------
# Core game objects
# -------------------------
class Paddle:
    def __init__(self, x, color):
        self.x = x
        self.y = HEIGHT // 2 - PADDLE_H // 2
        self.color = color
        self.vy = 0

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), PADDLE_W, PADDLE_H)

    def update(self, target_y, dt):
        # Move toward target_y with speed limit and slight smoothing
        center = self.y + PADDLE_H / 2
        dy = target_y - center
        desired_v = clamp(dy * 6.5, -PADDLE_MAX_SPEED, PADDLE_MAX_SPEED)
        # Smooth velocity (critically damped-ish)
        self.vy += (desired_v - self.vy) * min(1.0, 12.0 * dt)
        self.y += self.vy * dt
        if self.y < 0:
            self.y = 0
            self.vy = 0
        if self.y + PADDLE_H > HEIGHT:
            self.y = HEIGHT - PADDLE_H
            self.vy = 0

    def draw(self, surf, glow_layer):
        pygame.draw.rect(surf, self.color, self.rect, border_radius=10)
        pygame.draw.rect(glow_layer, (*self.color, 255), self.rect.inflate(14, 14), border_radius=16)

class Ball:
    def __init__(self):
        self.reset(direction=random.choice([-1, 1]))

        self.trail = deque(maxlen=TRAIL_LENGTH)

    def reset(self, direction=1):
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        ang = random.uniform(-0.35, 0.35) + (math.pi if direction < 0 else 0)
        self.vx = math.cos(ang) * BALL_SPEED
        self.vy = math.sin(ang) * BALL_SPEED
        self.size = BALL_SIZE
        self.color = C_YELLOW
        self.spin = 0.0

        self.trail = deque(maxlen=TRAIL_LENGTH)

    @property
    def rect(self):
        s = self.size
        return pygame.Rect(int(self.x - s / 2), int(self.y - s / 2), s, s)

    def update(self, dt):
        self.trail.appendleft((self.x, self.y))
        self.x += self.vx * dt
        self.y += self.vy * dt
        # top/bottom bounce
        if self.y - self.size / 2 <= 0:
            self.y = self.size / 2
            self.vy = abs(self.vy)
        elif self.y + self.size / 2 >= HEIGHT:
            self.y = HEIGHT - self.size / 2
            self.vy = -abs(self.vy)
        # mild continuous acceleration (optional)
        if BALL_ACCEL_TIME > 0:
            spd = math.hypot(self.vx, self.vy)
            spd = min(BALL_SPEED_MAX, spd + BALL_ACCEL_TIME)
            ang = math.atan2(self.vy, self.vx)
            self.vx = math.cos(ang) * spd
            self.vy = math.sin(ang) * spd

    def collide_paddle(self, paddle: Paddle):
        if self.rect.colliderect(paddle.rect):
            # collision point relative to paddle center (-1..1)
            rel = (self.y - (paddle.y + PADDLE_H / 2)) / (PADDLE_H / 2)
            rel = clamp(rel, -1.0, 1.0)

            # base angle (negative => up on screen because y grows downward)
            angle = rel * (math.pi / 3.25)  # max ~55 degrees

            # target speed after hit
            spd = min(BALL_SPEED_MAX, math.hypot(self.vx, self.vy) * BALL_ACCEL_HIT + 6.0)

            # decide bounce horizontal direction by paddle side (robust if self.vx is tiny)
            dir_x = 1 if paddle.x < WIDTH / 2 else -1

            # base components from angle (before spin)
            base_vy = math.sin(angle) * spd
            base_vx = math.cos(angle) * spd * dir_x

            # small "spin" / influence from paddle movement
            spin = paddle.vy * 0.3
            new_vy = base_vy + spin

            # Don't let spin invert the intended vertical direction (avoids stalls/inversions)
            if base_vy != 0 and sign(new_vy) != sign(base_vy):
                # keep same sign as base_vy and ensure reasonable magnitude
                new_vy = math.copysign(max(abs(base_vy) * 0.5, abs(spin)), base_vy)

            # Ensure a minimum vertical component so the ball doesn't run nearly horizontal
            min_vy = max(120.0, spd * 0.18)  # 120 px/s floor + proportional to speed
            if abs(new_vy) < min_vy:
                new_vy = math.copysign(min_vy, base_vy if base_vy != 0 else (spin or 1))

            # Recompute vx so total speed ≈ spd (preserve energy)
            new_vx = dir_x * math.sqrt(max(0.0, spd * spd - new_vy * new_vy))

            self.vx = new_vx
            self.vy = new_vy

            # push ball clearly outside the paddle to avoid repeated collisions
            push_out = 4  # pixels
            if dir_x > 0:
                self.x = paddle.x + PADDLE_W + self.size / 2 + push_out
            else:
                self.x = paddle.x - self.size / 2 - push_out

    def draw(self, surf, glow_layer):
        # draw trail
        for i, (tx, ty) in enumerate(self.trail):
            t = 1.0 - i / max(1, len(self.trail))
            a = int(200 * (t ** 1.5))
            r = max(2, int(self.size * (0.35 + 0.65 * t)))
            pygame.draw.circle(glow_layer, (*C_YELLOW, a), (int(tx), int(ty)), r + 6)
        pygame.draw.circle(surf, C_YELLOW, (int(self.x), int(self.y)), self.size // 2 + 1)
        pygame.draw.circle(glow_layer, (*C_YELLOW, 255), (int(self.x), int(self.y)), self.size // 2 + 7)

# -------------------------
# AI Controllers
# -------------------------
class BaseAI:
    def __init__(self, side: str):
        self.side = side  # 'left' or 'right'
        self.reaction = 0.12  # seconds of reaction smoothing
        self.target_y = HEIGHT / 2

    def update(self, ball: Ball, my_paddle: Paddle, opp_paddle: Paddle, dt):
        raise NotImplementedError

class PredictiveAI(BaseAI):
    """Predicts the y-position where the ball will cross the paddle's x and aims there.
    Simulates wall bounces. Adds slight noise and delay to be beatable."""
    def __init__(self, side):
        super().__init__(side)
        self.noise = 6.0
        self.lookahead_limit = 2.5  # seconds

    def _predict_intercept_y(self, ball: Ball, target_x):
        # Simulate the ball's future path (x, y, vx, vy) with wall bounces
        x, y = ball.x, ball.y
        vx, vy = ball.vx, ball.vy
        dt = 1.0 / FPS
        t = 0.0
        for _ in range(int(self.lookahead_limit * FPS)):
            x += vx * dt
            y += vy * dt
            if y - BALL_SIZE / 2 <= 0:
                y = BALL_SIZE / 2
                vy = abs(vy)
            elif y + BALL_SIZE / 2 >= HEIGHT:
                y = HEIGHT - BALL_SIZE / 2
                vy = -abs(vy)
            # Reached the x?
            if (target_x - x) * vx <= 0:  # crossed
                # linear interpolation to target_x
                if vx != 0:
                    ratio = (target_x - x) / vx
                    y_cross = y + vy * ratio
                else:
                    y_cross = y
                return clamp(y_cross, BALL_SIZE/2, HEIGHT - BALL_SIZE/2)
            t += dt
        return HEIGHT / 2  # fallback

    def update(self, ball: Ball, my_paddle: Paddle, opp_paddle: Paddle, dt):
        target_x = my_paddle.x + (PADDLE_W if self.side == 'left' else 0)
        # Only predict when ball is moving towards this side
        if (self.side == 'left' and ball.vx < 0) or (self.side == 'right' and ball.vx > 0):
            py = self._predict_intercept_y(ball, target_x)
            py += random.uniform(-self.noise, self.noise)
        else:
            # Centering behavior when ball goes away
            py = HEIGHT / 2 + math.sin(pygame.time.get_ticks() * 0.001) * 40.0
        # Smooth target with reaction delay
        self.target_y += (py - self.target_y) * clamp(dt / self.reaction, 0.0, 1.0)
        # Convert desired center to paddle-center target
        my_paddle.update(self.target_y, dt)

class SmoothTrackAI(BaseAI):
    """Tracks the ball with smooth bias and slight prediction depending on distance."""
    def __init__(self, side):
        super().__init__(side)
        self.bias = random.uniform(-18.0, 18.0)
        self.pred_scale = 0.22  # how much of future position to consider
        self.reaction = 0.10

    def update(self, ball: Ball, my_paddle: Paddle, opp_paddle: Paddle, dt):
        # Predict a bit into the future proportional to distance
        dist_x = abs((my_paddle.x + PADDLE_W/2) - ball.x)
        t_predict = clamp(dist_x / abs(ball.vx or 1) * self.pred_scale, 0.0, 0.35)
        py = ball.y + ball.vy * t_predict + self.bias
        self.target_y += (py - self.target_y) * clamp(dt / self.reaction, 0.0, 1.0)
        my_paddle.update(self.target_y, dt)

# -------------------------
# Game
# -------------------------
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Neon Pong — AI vs AI (pygame)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont("arialroundedmtbold", 64)
        self.font_med = pygame.font.SysFont("arialroundedmtbold", 28)
        self.font_small = pygame.font.SysFont("arial", 18)

        # Fixed layers
        self.scanlines = make_scanlines(WIDTH, HEIGHT, spacing=4, alpha=20)
        self.vignette = make_vignette(WIDTH, HEIGHT, strength=180)

        # Game state
        self.left = Paddle(PADDLE_MARGIN, C_CYAN)
        self.right = Paddle(WIDTH - PADDLE_MARGIN - PADDLE_W, C_MAGENTA)
        self.ball = Ball()

        self.fx = FX()

        self.score_l = 0
        self.score_r = 0

        # AI controllers
        self.ai_left = PredictiveAI('left')
        self.ai_right = SmoothTrackAI('right')

        self.running = True
        self.round_over_delay = 1.0
        self.timer = 0.0
        self.state = "play"  # play, point_pause, game_over

        # Prebuild a glow layer (offscreen)
        self.glow_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    def reset_round(self, direction):
        self.ball.reset(direction=direction)
        self.ai_left.target_y = HEIGHT / 2
        self.ai_right.target_y = HEIGHT / 2

    def score(self, who):
        if who == 'left':
            self.score_l += 1
            self.fx.burst((WIDTH * 0.25, HEIGHT / 2), PARTICLE_COUNT_SCORE, C_CYAN)
        else:
            self.score_r += 1
            self.fx.burst((WIDTH * 0.75, HEIGHT / 2), PARTICLE_COUNT_SCORE, C_MAGENTA)

        if self.score_l >= WIN_SCORE or self.score_r >= WIN_SCORE:
            self.state = "game_over"
            self.timer = 0.0
        else:
            self.state = "point_pause"
            self.timer = 0.0

    def draw_scores(self, surface):
        s_l = self.font_big.render(str(self.score_l), True, C_CYAN)
        s_r = self.font_big.render(str(self.score_r), True, C_MAGENTA)
        surface.blit(s_l, (WIDTH * 0.25 - s_l.get_width() / 2, 24))
        surface.blit(s_r, (WIDTH * 0.75 - s_r.get_width() / 2, 24))

    def handle_collisions(self):
        pre_vx = self.ball.vx
        if self.ball.rect.colliderect(self.left.rect):
            self.ball.collide_paddle(self.left)
            self.fx.burst((self.left.x + PADDLE_W, self.ball.y), PARTICLE_COUNT_HIT, C_CYAN)
        elif self.ball.rect.colliderect(self.right.rect):
            self.ball.collide_paddle(self.right)
            self.fx.burst((self.right.x, self.ball.y), PARTICLE_COUNT_HIT, C_MAGENTA)

        # Net streak when speed increases
        if sign(pre_vx) != sign(self.ball.vx):
            pass  # already handled

    def update(self, dt):
        if self.state == "play":
            # Update AI paddles
            self.ai_left.update(self.ball, self.left, self.right, dt)
            self.ai_right.update(self.ball, self.right, self.left, dt)

            # Update ball
            self.ball.update(dt)
            self.handle_collisions()

            # Check scoring
            if self.ball.x < -BALL_SIZE:
                self.score('right')
            elif self.ball.x > WIDTH + BALL_SIZE:
                self.score('left')

        elif self.state == "point_pause":
            self.timer += dt
            if self.timer >= self.round_over_delay:
                # Serve toward the player who conceded
                direction = -1 if self.ball.x > WIDTH / 2 else 1
                self.reset_round(direction)
                self.state = "play"

        elif self.state == "game_over":
            self.timer += dt

        # Update FX
        self.fx.update(dt)

    def draw_background(self, surface):
        surface.fill(C_BG)
        # Subtle grid / lines
        for i in range(0, WIDTH, 48):
            pygame.draw.line(surface, C_LINES, (i, 0), (i, HEIGHT))
        for j in range(0, HEIGHT, 48):
            pygame.draw.line(surface, C_LINES, (0, j), (WIDTH, j))
        draw_center_dashed_line(surface, color=(60, 80, 90), dash_len=18, gap=14, width=4)

    def draw_glow(self):
        # Simple multi-pass blur via repeated alpha expand
        # (Emulates glow; performance-friendly for our scale)
        # Already drawn bright shapes into glow_layer; now blur it a bit
        # For speed, just scale down then up which approximates blur.
        scale = clamp(1.0 + GLOW_STRENGTH * 0.15, 1.0, 2.8)
        w = int(WIDTH / scale)
        h = int(HEIGHT / scale)
        small = pygame.transform.smoothscale(self.glow_layer, (w, h))
        big = pygame.transform.smoothscale(small, (WIDTH, HEIGHT))
        return big

    def draw(self):
        base = self.screen
        self.draw_background(base)

        # Clear glow layer
        self.glow_layer.fill((0, 0, 0, 0))

        # Draw actors
        self.left.draw(base, self.glow_layer)
        self.right.draw(base, self.glow_layer)
        self.ball.draw(base, self.glow_layer)

        # FX
        self.fx.draw(self.glow_layer)

        # Composite glow
        glow = self.draw_glow()
        base.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)

        # UI
        self.draw_scores(base)

        if self.state == "game_over":
            winner = "Left AI" if self.score_l > self.score_r else "Right AI"
            col = C_CYAN if self.score_l > self.score_r else C_MAGENTA
            t1 = self.font_big.render(f"{winner} wins!", True, col)
            t2 = self.font_med.render("Press [R] to restart  •  [ESC] to quit", True, C_WHITE)
            base.blit(t1, (WIDTH / 2 - t1.get_width() / 2, HEIGHT / 2 - 48))
            base.blit(t2, (WIDTH / 2 - t2.get_width() / 2, HEIGHT / 2 + 18))

        # Overlays
        base.blit(self.scanlines, (0, 0), special_flags=pygame.BLEND_MULT)
        base.blit(self.vignette, (0, 0))

        # Footer
        foot = self.font_small.render("Neon Pong — PredictiveAI vs SmoothTrackAI  •  pygame", True, (170, 190, 200))
        base.blit(foot, (WIDTH / 2 - foot.get_width() / 2, HEIGHT - 26))

        pygame.display.flip()

    def handle_input(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                elif e.key == pygame.K_r and self.state == "game_over":
                    self.score_l = self.score_r = 0
                    self.state = "play"
                    self.reset_round(direction=random.choice([-1, 1]))

    def run(self):
        self.reset_round(direction=random.choice([-1, 1]))
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_input()
            self.update(dt)
            self.draw()
        pygame.quit()

def main():
    Game().run()

if __name__ == "__main__":
    main()
