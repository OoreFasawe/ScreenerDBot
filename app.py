# Discord Bot for Stock Screener - Sends daily stock updates as images
import sys
import os
import io
import uvicorn
from fastapi import FastAPI, Request
import pandas as pd
import matplotlib.pyplot as plt
import nextcord
from nextcord.ext import commands, tasks
import json
import requests
import logging

# ========== CONFIGURATION ==========
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DATA_ENDPOINT = "https://get-top-rvol-stocks-964914644779.europe-west1.run.app"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Discord bot intents (permissions)
intents = nextcord.Intents.default()
client = nextcord.Client(intents=intents)
app = FastAPI()

# ========== TABLE COLUMN CONFIGURATION ==========
COLUMNS_TO_KEEP = [
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ATR",
    "RVol",
    "1D Ret",
    "OTC Ret"
]

def filter_stock_columns(stocks, columns):
    """
    Returns a new list of dicts, each containing only the specified columns.
    """
    filtered = []
    for stock in stocks:
        filtered.append({col: stock.get(col, "") for col in columns})
    return filtered

# ========== PAYLOAD CONVERSION ==========
def convert_payload_to_list(data):
    """
    Converts a dict-of-dicts keyed by ticker to a list of dicts with mapped/renamed columns.
    """
    stocks = []
    for ticker, info in data.items():
        stock = {'Ticker': ticker}
        stock.update(info)
        # Map/rename columns for table
        stock["Volume"] = info.get("14D_Avg_Volume", "")
        stock["RVol"] = info.get("Relative Volume", "")
        stock["1D Ret"] = info.get("Prev day % return", "")
        stock["OTC Ret"] = info.get("Prev day % return Open To Close", "")
        stocks.append(stock)
    return stocks

# ========== TABLE GENERATION ==========
def style_table_cells(table, col_labels):
    """
    Apply custom styling to table cells: header colors, bold, and value-based coloring.
    """
    for (row, col), cell in table.get_celld().items():
        # Header row
        if row == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold')
            # Color headers
            if col > 0 and col_labels[col-1] == "High":
                cell.set_facecolor("#ffe600")  # yellow
                cell.set_text_props(color='black')
            elif col > 0 and col_labels[col-1] == "Low":
                cell.set_facecolor("#ffb3b3")  # light red
                cell.set_text_props(color='black')
            elif col > 0 and col_labels[col-1] == "Close":
                cell.set_facecolor("#b3d1ff")  # light blue
                cell.set_text_props(color='black')
            else:
                cell.set_facecolor("#333333")
                cell.set_text_props(color='white')
        # Row labels (Ticker)
        elif col == 0:
            cell.set_fontsize(12)
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor("#222222")
        # Data cells
        else:
            cell.set_fontsize(11)
            cell.set_facecolor("#222222")
            cell.set_text_props(color='white')
            # Color returns green/red
            if col > 0 and col_labels[col-1] in ["1D Ret", "OTC Ret", "Prev 1D Ret COC", "Prev 1D Ret OTC"]:
                try:
                    val = float(cell.get_text().get_text())
                    if val > 0:
                        cell.set_text_props(color='#00ff00')  # green
                    elif val < 0:
                        cell.set_text_props(color='#ff6666')  # red
                except Exception:
                    pass
            # Color High/Low/Close values
            if col > 0 and col_labels[col-1] == "High":
                cell.set_text_props(color='#ffe600')
            if col > 0 and col_labels[col-1] == "Low":
                cell.set_text_props(color='#ff6666')
            if col > 0 and col_labels[col-1] == "Close":
                cell.set_text_props(color='#b3d1ff')

def generate_table_image(stocks):
    """
    Generate a PNG image of the stock table using pandas and matplotlib.
    Returns a BytesIO buffer containing the image.
    """
    df = pd.DataFrame(stocks)
    df.set_index("Ticker", inplace=True)
    cell_text = df.values.tolist()
    col_labels = list(df.columns)
    row_labels = list(df.index)
    fig, ax = plt.subplots(figsize=(len(col_labels) * 0.7, len(df) * 0.35), dpi=300)
    ax.axis('off')
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc='center',
        cellLoc='center',
        rowLoc='center'
    )
    style_table_cells(table, col_labels)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width(col=list(range(len(col_labels)+1)))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

# ========== DISCORD BOT LOGIC ==========
@app.post("/notify")
async def notify(request: Request):
    data = await request.json()
    
    stocks = data.get("stocks", [])
    stocks = convert_payload_to_list(stocks)
    # Filter columns here
    stocks = filter_stock_columns(stocks, COLUMNS_TO_KEEP)
    # Sort by RVol descending
    stocks.sort(key=lambda x: float(x.get("RVol") or 0), reverse=True)
    image_buf = generate_table_image(stocks)
    await client.wait_until_ready()
    channels = client.get_all_channels()
    for channel in channels:
        if channel is not None and isinstance(channel, nextcord.TextChannel):
            logger.info(f"Sending screener image to channel: {channel.name} (ID: {channel.id})")
            await channel.send(file=nextcord.File(image_buf, 'screener.png'))
    logger.info("All messages sent. Closing bot.")
    return {"status": "Image sent"}

def start_bot():
    uvicorn.run(app, host="0.0.0.0", port=8080)

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    import threading
    threading.Thread(target=start_bot, daemon=True).start()
    client.run(DISCORD_TOKEN)

