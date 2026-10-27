"""Learning sandbox: replay the bot's own trade logic over history.

Every day of every coin becomes a *situation* (a vector of what the market
looked like at that close) and every strategy that would have acted on it
becomes a simulated trade with its real outcome. The two together answer the
question the plan card cannot: "what happened after situations like this one?"

Nothing here talks to Telegram. The data lives in its own SQLite file
(``runtime/learning/learning.db``) so a replay never touches user data.
"""
