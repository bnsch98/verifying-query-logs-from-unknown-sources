import traceback


input_file_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas/orcas-doctrain-queries.tsv"
# Pfad zur bereinigten CSV-Datei
cleaned_file_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas/orcas-doctrain-queries-cleaned.csv"


# Pfad zur Log-Datei
log_file_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-schneg/orcas/processing_log.txt"

# Schritt 1: Datei bereinigen


def clean_line(line):
    try:
        # Trenne die Zeile in zwei Teile: ID und String
        parts = line.split("\t")
        if len(parts) == 2:
            id_part = parts[0].strip()
            string_part = parts[1].strip()
        else:
            parts = line.split(" ", 1)
            id_part = parts[0].strip()
            string_part = parts[1].strip()

        # Ersetze das Trennzeichen durch ein Komma und gebe die bereinigte Zeile zurück
        return f"{id_part},{string_part}\n"
    except Exception as e:
        # Logge den Fehler und die problematische Zeile
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Fehler beim Verarbeiten der Zeile: {line}\n")
            log_file.write(f"Fehlermeldung: {traceback.format_exc()}\n")
        return None


cnt = 0

with open(input_file_path, 'r', encoding='utf-8') as infile, open(cleaned_file_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            cnt += 1
            cleaned_line = clean_line(line)
            if cleaned_line:
                outfile.write(cleaned_line)

            if cnt % 300000 == 0:
                print(f"{cnt} Zeilen verarbeitet")
                with open(log_file_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"{cnt} Zeilen verarbeitet\n")
        except Exception as e:
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(
                    f"Ein schwerwiegender Fehler ist aufgetreten: {traceback.format_exc()}\n")

print("Dateibereinigung abgeschlossen.")
