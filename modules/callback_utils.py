"""Callback-query helpers shared by handlers and payments."""

import logging
from collections import OrderedDict

from telegram.error import BadRequest

logger = logging.getLogger(__name__)

_MAX_TRACKED = 2048
_answered: "OrderedDict[object, None]" = OrderedDict()


async def answer_once(query, text=None, show_alert: bool = False) -> bool:
    """Answer a callback query at most once.

    Telegram accepts a single answer per query and rejects any further one
    with BadRequest; several handler branches answer with their own toast
    after a generic answer, which used to abort the branch mid-way.
    """
    if query is None or query.id in _answered:
        return False
    _answered[query.id] = None
    while len(_answered) > _MAX_TRACKED:
        _answered.popitem(last=False)
    try:
        await query.answer(text=text, show_alert=show_alert)
        return True
    except BadRequest as error:
        logger.debug("Callback answer rejected: %s", error)
        return False
