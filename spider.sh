#two days ago in YYYY-MM-DD format
datebeg=$(date -d "2 days ago" '+%Y-%m-%d')
datebeg=$(date -d "yesterday" '+%Y-%m-%d')
maxpapers=4
#today
dateend=$(date '+%Y-%m-%d')
topic="neural+network+potentials"

mkdir -p input
for i in $(curl "https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=$topic&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date=$datebeg&date-to_date=$dateend&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first" 2>&1 | grep -Eo "https://arxiv.org/pdf/[0-9]+\.[0-9]+" | head -n $maxpapers)
do
    doi=$(echo $i | grep -Eo "[0-9]+\.[0-9]+")
    #Get title of the paper doi without using doi.org
    title=$(curl -s "https://arxiv.org/abs/$doi" | grep -m 1 '<title>' | sed -n 's/.*\]\(.*\)<\/title>/\1/p')
    echo "Getting $doi : $title"
    curl -s $i.pdf > input/$doi.pdf
    pdftotext input/$doi.pdf
done
