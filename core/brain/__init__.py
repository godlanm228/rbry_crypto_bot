"""The brain: one model over every coin-day, trained on what really happened.

dataset.py    one row per coin per day from market.db: what was known at that
              close, and how a fixed trade bought at the next open ended
model.py      gradient boosting on those rows, plus the calibration that makes
              "6 of 10" mean 6 of 10
walkforward.py  the exam: refit on the past only, grade on months it never saw

Research first: nothing here is wired into the bot until the exam says the
brain beats entering at random.
"""
