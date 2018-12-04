echo "Start!"

while read line; 
do
 alpha-vantage-cli --type=daily --symbol="$line";
 --api-key=82KAKRXYKTSMEUPU
 --out="$line"_daily.csv --output-data-size="full"
done < test.txt