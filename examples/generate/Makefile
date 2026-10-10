HIKEC = go run ../../../cmd/hikec
TARGET = windows
OUTPUT = generate.exe
MODEL = Qwen3-0.6B-UD-Q4_K_XL.gguf
MODEL_URL = https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-UD-Q4_K_XL.gguf?download=true
CURL = curl.exe

.PHONY: all download emit-ir build run clean

all: download build

download: $(MODEL)

$(MODEL):
	$(CURL) -L --fail --retry 3 -o $@ "$(MODEL_URL)"

emit-ir:
	$(HIKEC) emit-ir -target $(TARGET) -o main.ll main.hike

build: $(OUTPUT)

$(OUTPUT): main.hike hike.mod ../../hike.mod
	$(HIKEC) build -target $(TARGET) main.hike -o $(OUTPUT)

run: download build
	./$(OUTPUT)

clean:
	rm -f main.ll $(OUTPUT) *.dll
