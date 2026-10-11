"""Small online-only example. Install racing-lab[rl] before running."""

from racing_lab.rl.training import train_sac

if __name__ == "__main__":
    result = train_sac("results/sac-example", track="monza", steps=3000, seed=0)
    print(result)
