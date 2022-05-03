from scores import User

# Example to get user scores from User class

zombie = User('0xE7D9e39e0AC47F6dB3871a3E648880f79e8eEF44')
zombie.load_all()
scores = zombie.get_scores()

print(scores)

