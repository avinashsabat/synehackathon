from fastapi import FastAPI, HTTPException
import requests
import uvicorn  # Make sure uvicorn is installed

app = FastAPI(title="VIN Decoder API", version="1.0")

NHTSA_API_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}?format=json"

@app.get("/decode-vin/{vin}")
def decode_vin(vin: str):
    vin = vin.upper()
    if len(vin) != 17:
        raise HTTPException(status_code=400, detail="VIN must be 17 characters long")

    response = requests.get(NHTSA_API_URL.format(vin=vin))

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Failed to reach VIN decoding service")

    data = response.json()
    results = {item["Variable"]: item["Value"] for item in data.get("Results", []) if item["Value"]}
    return {
        "vin": vin,
        "decoded": results
    }

# Run the app directly if this is the main file
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
