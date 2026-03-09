import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

def process_data(input_file, output_folder):
    "" "Zavatazhennja ta peredobrobka danyh" ""
    print(f"Zavatazhuju dani z {input_file}...")
    
    # 2. Zavantazhennya danyh za dopomohoyu pandas
    df = pd.read_csv(input_file)

    # 3. Feature engineering
    print("Vikonuju feature engineering...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month

    # Vydalennya zai'vyh kolonok dlya unyknennya vytoku danyh
    cols_to_drop = ['datetime', 'casual', 'registered']
    # Pereviryayemo chy ye ci kolonki pered vydalennjam (shob ne bulo pomylky pry zapusku na test.csv)
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 4. Rozpodil danyh na train ta test
    # Perevirka chy my mayemo cikovy zminnu count (dlya raw/train.csv) abo tse raw/test.csv de yiyi nema
    if 'count' in df.columns:
        print("Rozbyvayu dani na train ta test...")
        # X ta y
        X = df.drop(columns=['count'])
        y = df['count']
        
        # Oskilki my musimo zberegti train.csv/test.csv razom z ih count v odnomu df, 
        # my rozbivayemo sam dataframe
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # 5. Zberezhennya obroblenyh danyh u papku vyvodu
        os.makedirs(output_folder, exist_ok=True)
        
        train_path = os.path.join(output_folder, "train.csv")
        test_path = os.path.join(output_folder, "test.csv")
        
        print(f"Zbregayu obrobleni dani u {train_path} ta {test_path}...")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
    else:
        # Yakshcho ce test.csv (z data/raw), my prosto yogo obrobliayemo ta zberigayemo
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "test_features.csv")
        print(f"Zbregayu obroblenu testovu vybirku u {out_path}...")
        df.to_csv(out_path, index=False)
        
    print("Gotovo!")

if __name__ == "__main__":
    # 1. Скрипт має приймати два аргументи через sys.argv
    if len(sys.argv) != 3:
        print("Vykorysanya: python src/prepare.py <shlyah_do_vhidnogo_failu> <shlyah_do_papky_vyvodu>")
        sys.exit(1)
        
    input_file_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    
    process_data(input_file_path, output_folder_path)
