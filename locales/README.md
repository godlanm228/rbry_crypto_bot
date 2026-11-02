# Translation Guide for RBRY CryptoBot

## Overview

This directory contains the multi-language translation system for RBRY CryptoBot. The bot supports 4 languages:
- **Russian (ru)** - Master/source language
- **Ukrainian (uk)**
- **English (en)**
- **German (de)**

## File Structure

```
locales/
├── __init__.py             # Package initialization
├── translator.py           # Translation engine with caching
├── strings/
│   ├── ru.json            # Russian (master)
│   ├── uk.json            # Ukrainian
│   ├── en.json            # English
│   └── de.json            # German
└── README.md              # This file
```

## Translation File Format

Translations are organized hierarchically in JSON format:

```json
{
  "category": {
    "subcategory": {
      "key": "Translation text with {parameter}"
    }
  }
}
```

### Categories

- **common** - Shared buttons and labels (back, main menu, loading, error)
- **menu** - Main menu items
- **welcome** - Start/welcome messages
- **portfolio** - Portfolio management
- **stats** - Statistics display
- **movers** - Top movers
- **analysis** - Technical analysis
- **settings** - Settings menu
- **tier** - Access tier information
- **premium** - Premium/payment features
- **admin** - Admin panel (owner only)
- **alerts** - Alert system
- **news** - News module
- **help** - Help commands
- **errors** - Error messages

## Parameter Placeholders

Use curly braces `{}` for dynamic values:

### Simple Parameters
```json
{
  "message": "Welcome {user_name}!"
}
```

### Formatted Parameters

**Numbers with decimals:**
```json
{
  "price": "Price: ${price:.2f}"  // 2 decimal places
}
```

**Percentages:**
```json
{
  "change": "Change: {change:+.2f}%"  // +/- sign, 2 decimals
}
```

**Large numbers:**
```json
{
  "coin_amount": "{amount:.6f} {coin}"  // 6 decimal places for crypto amounts
}
```

### Common Parameters

- `{coin}` - Coin symbol (BTC, ETH, etc.)
- `{amount}` - Numeric amount (use :.6f for crypto)
- `{price}` - Price (use :.2f or :.4f)
- `{change}` - Percentage change (use :+.2f for +/- sign)
- `{user_id}` - User ID
- `{time}` - Timestamp
- `{count}` - Count of items
- `{tier}` - Access tier name
- `{stars}` - Telegram Stars amount

## Formatting Rules

### 1. Preserve Emojis
Keep all emojis exactly as in Russian:
```json
{
  "menu.portfolio": "💼 Portfolio"  // ✅ Keep emoji
}
```

### 2. Preserve Markdown
Keep markdown formatting markers:
```json
{
  "title": "**STATISTICS**"  // ✅ Keep **bold**
}
```

### 3. Preserve HTML
Keep HTML tags where present:
```json
{
  "settings.title": "⚙️ <b>Settings</b>"  // ✅ Keep <b> tags
}
```

### 4. Preserve Line Breaks
Keep `\n` newline characters:
```json
{
  "multiline": "Line 1\nLine 2\nLine 3"  // ✅ Keep \n
}
```

### 5. Preserve Special Characters
- Dividers: `═`, `━`, `─`, `├`, `└`, `↳`
- Quotes: Keep backticks for code: `<code>text</code>`

## Translation Guidelines

### Do:
- ✅ **Translate meaning, not words** - Adapt to natural phrasing in target language
- ✅ **Preserve technical terms** - Keep "Portfolio", "RSI", "MACD" untranslated
- ✅ **Keep crypto terminology consistent**:
  - Bitcoin -> Bitcoin (same in all languages)
  - Wallet -> Portfolio/Portfel/Hamanets/Geldbörse
  - Trading -> Trading/Treiding/Treidinh/Handel
- ✅ **Maintain tone** - Bot is friendly but professional
- ✅ **Test formatting** - Check that strings display correctly with emojis and markdown

### Don't:
- ❌ **Don't translate parameter names** - Keep `{coin}`, `{amount}` as-is
- ❌ **Don't change formatting** - Keep :.2f, :+.1f precision specifiers
- ❌ **Don't add/remove emojis** - Keep emoji usage consistent
- ❌ **Don't translate technical indicators** - RSI, MACD, SMA stay in English
- ❌ **Don't translate /commands** - Keep /start, /help, /portfolio as-is

## Cryptocurrency Terms

Standard translations:

| English | Russian | Ukrainian | German |
|---------|---------|-----------|--------|
| Portfolio | Портфель | Портфель | Portfolio |
| Analysis | Анализ | Аналіз | Analyse |
| Statistics | Статистика | Статистика | Statistik |
| Signals | Сигналы | Сигнали | Signale |
| Premium | Премиум | Преміум | Premium |
| Subscription | Подписка | Підписка | Abonnement |
| Wallet | Кошелек | Гаманець | Geldbörse |

## Testing Your Translation

### 1. Validate JSON Syntax
```bash
python -m json.tool locales/strings/en.json
```

### 2. Run Unit Tests
```bash
python -m pytest tests/test_translator.py -v
```

### 3. Test in Bot
```python
from locales.translator import t

# Test simple string
print(t("menu.portfolio", lang="en"))

# Test with parameters
print(t("portfolio.coin_entry", lang="en", coin="BTC", amount=0.5, price=50000))
```

## Translation Workflow

1. **Extract strings** - Start from ru.json (master)
2. **Translate** - Copy structure to target language file
3. **Validate** - Check JSON syntax
4. **Test** - Run unit tests
5. **Review** - Native speaker review
6. **Deploy** - Bot automatically picks up changes on restart

## Quality Checklist

Before submitting translations, verify:

- [ ] All keys from ru.json are present
- [ ] No missing parameter placeholders `{variable}`
- [ ] Formatting markers preserved (`**bold**`, `<b>`, `\n`)
- [ ] Emojis match source language
- [ ] Crypto terminology is accurate
- [ ] Native speaker reviewed
- [ ] JSON syntax is valid
- [ ] Unit tests pass
- [ ] Tested in bot with real data

## Examples

### Good Translation Example (English)

**Russian:**
```json
{
  "portfolio": {
    "title": "💼 **ПОРТФЕЛЬ:**",
    "empty": "📭 Портфель пуст.",
    "coin_entry": "💰 {coin}: {amount:.6f} @ ${price:.2f}"
  }
}
```

**English:**
```json
{
  "portfolio": {
    "title": "💼 **PORTFOLIO:**",
    "empty": "📭 Portfolio is empty.",
    "coin_entry": "💰 {coin}: {amount:.6f} @ ${price:.2f}"
  }
}
```

✅ **What's correct:**
- Emojis preserved (💼, 📭, 💰)
- Markdown preserved (`**PORTFOLIO:**`)
- Parameters preserved (`{coin}`, `{amount:.6f}`, `{price:.2f}`)
- Natural English phrasing

### Bad Translation Example

**Russian:**
```json
{
  "analysis": {
    "analyzing_portfolio": "🧠 Анализирую портфель... [{time}]"
  }
}
```

**Bad English:**
```json
{
  "analysis": {
    "analyzing_portfolio": "🧠 I analyze portfolio... [time]"
  }
}
```

❌ **What's wrong:**
- Missing parameter braces: `[time]` instead of `[{time}]`
- Unnatural phrasing: "I analyze" instead of "Analyzing"

**Good English:**
```json
{
  "analysis": {
    "analyzing_portfolio": "🧠 Analyzing portfolio... [{time}]"
  }
}
```

## Getting Help

If you have questions about translations:

1. Check this README
2. Look at ru.json for context
3. Review existing translations in en.json
4. Test with unit tests
5. Ask maintainer if unsure

## Contributing

To contribute translations:

1. Fork repository
2. Edit language file (uk.json, en.json, or de.json)
3. Run validation: `python -m json.tool locales/strings/YOUR_LANG.json`
4. Run tests: `pytest tests/test_translator.py`
5. Submit pull request
6. Native speaker review required

## Technical Details

### Caching
- All translations loaded into memory on bot startup
- O(1) lookup time (dictionary access)
- Reload without restart: `/reload_translations` command (admin only)

### Fallback Chain
If a key is missing:
1. Try requested language
2. Fallback to Russian (master)
3. Fallback to English
4. Return `[key]` if still not found

### Performance
- ~14KB per language file
- ~56KB total for 4 languages
- <0.1ms per translation lookup
- <1MB total memory footprint

## License

Translations follow the same license as the main project.

