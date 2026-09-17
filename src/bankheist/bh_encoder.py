
import jax.numpy as jnp


WIDTH, HEIGHT = 160.0, 210.0
FUEL_CAPACITY = 10200.0  # confirmed from BankHeistConstants source


def bankheist_features(obs) -> jnp.ndarray:
    """
    Flat feature vector for a policy network. Mirrors the Pong/Ms. Pac-Man
    pattern: raw object-relative features (neural-friendly) plus a couple of
    named symbolic relations (nearest-danger, nearest-target) that a purely
    positional encoding would make the network re-derive from scratch.
    """
    px, py = obs.player.x.astype(jnp.float32), obs.player.y.astype(jnp.float32)

    # --- Enemies (3 police cars): relative offset + active flag each ---
    enemy_dx = (obs.enemies.x.astype(jnp.float32) - px) / WIDTH
    enemy_dy = (obs.enemies.y.astype(jnp.float32) - py) / HEIGHT
    enemy_active = obs.enemies.active.astype(jnp.float32)
    enemy_feats = jnp.stack([enemy_dx, enemy_dy, enemy_active], axis=-1).reshape(-1)  # (9,)

    # --- Banks (3): relative offset + active flag each ---
    bank_dx = (obs.banks.x.astype(jnp.float32) - px) / WIDTH
    bank_dy = (obs.banks.y.astype(jnp.float32) - py) / HEIGHT
    bank_active = obs.banks.active.astype(jnp.float32)
    bank_feats = jnp.stack([bank_dx, bank_dy, bank_active], axis=-1).reshape(-1)  # (9,)

    # --- Dynamite: relative offset + active flag ---
    dyn_dx = (obs.dynamite.x.astype(jnp.float32) - px) / WIDTH
    dyn_dy = (obs.dynamite.y.astype(jnp.float32) - py) / HEIGHT
    dyn_active = obs.dynamite.active.astype(jnp.float32)

    # --- Symbolic relations: nearest ACTIVE enemy / bank, masking out inactive slots ---
    enemy_dist = jnp.sqrt(enemy_dx**2 + enemy_dy**2)
    enemy_dist_masked = jnp.where(enemy_active > 0, enemy_dist, jnp.inf)
    nearest_enemy_dist = jnp.min(enemy_dist_masked)
    nearest_enemy_dist = jnp.where(jnp.isinf(nearest_enemy_dist), 1.0, nearest_enemy_dist)

    bank_dist = jnp.sqrt(bank_dx**2 + bank_dy**2)
    bank_dist_masked = jnp.where(bank_active > 0, bank_dist, jnp.inf)
    nearest_bank_idx = jnp.argmin(bank_dist_masked)
    nearest_bank_dx = jnp.where(bank_active[nearest_bank_idx] > 0, bank_dx[nearest_bank_idx], 0.0)
    nearest_bank_dy = jnp.where(bank_active[nearest_bank_idx] > 0, bank_dy[nearest_bank_idx], 0.0)

    # --- Scalars: fuel ratio (normalized by the real FUEL_CAPACITY=10200.0,
    # confirmed from source), fuel_refill flag (already 0/1), lives ---
    fuel_norm = obs.fuel.astype(jnp.float32) / FUEL_CAPACITY
    fuel_refill_flag = obs.fuel_refill.astype(jnp.float32)
    lives = obs.lives.astype(jnp.float32)

    return jnp.concatenate([
        jnp.array([px / WIDTH, py / HEIGHT]),          # 2
        enemy_feats,                                     # 9
        bank_feats,                                       # 9
        jnp.array([dyn_dx, dyn_dy, dyn_active]),          # 3
        jnp.array([nearest_enemy_dist, nearest_bank_dx, nearest_bank_dy]),  # 3
        jnp.array([fuel_norm, fuel_refill_flag, lives]),  # 3
    ])  # total: 29-dim


if __name__ == "__main__":
    # Mock an observation matching the REAL confirmed structure exactly --
    # no sprites needed for this, since ObjectObservation is a plain pytree.
    from jaxatari.environment import ObjectObservation

    player = ObjectObservation.create(x=jnp.array(80), y=jnp.array(100),
                                       width=jnp.array(8), height=jnp.array(8),
                                       active=jnp.array(1))
    enemies = ObjectObservation.create(
        x=jnp.array([90, 0, 0]), y=jnp.array([110, 0, 0]),
        width=jnp.full((3,), 8), height=jnp.full((3,), 8),
        active=jnp.array([1, 0, 0]),  # only 1 of 3 police currently active
    )
    banks = ObjectObservation.create(
        x=jnp.array([60, 130, 0]), y=jnp.array([90, 40, 0]),
        width=jnp.full((3,), 8), height=jnp.full((3,), 8),
        active=jnp.array([1, 1, 0]),  # 2 of 3 banks currently active
    )
    dynamite = ObjectObservation.create(x=jnp.array(0), y=jnp.array(0),
                                         width=jnp.array(8), height=jnp.array(8),
                                         active=jnp.array(0))

    class FakeObs:
        pass
    obs = FakeObs()
    obs.player, obs.enemies, obs.banks, obs.dynamite = player, enemies, banks, dynamite
    obs.fuel = jnp.array(1500)
    obs.fuel_refill = jnp.array(0)
    obs.lives = jnp.array(3)

    feats = bankheist_features(obs)
    print("feature vector shape:", feats.shape)
    print("features:", feats)
