exampleurl = "https://psl.noaa.gov/cgi-bin/mddb2/plot.pl?doplot=0&varID=2684&fileID=118848&itype=0&variable=mslp&levelType=Sea%20Level&level_units=&level=Sea%20Level&timetype=4x&fileTimetype=4x&createAverage=1&year1=2003&month1=1&day1=1&hr1=00%20Z&year2=2003&month2=12&day2=31&hr2=18%20Z&region=Custom&area_north=60&area_west=-150&area_east=-90&area_south=20&centerLat=0.0&centerLon=270.0"
urlformat = [
    "https://psl.noaa.gov/cgi-bin/mddb2/plot.pl?doplot=0&varID=", 
    "&itype=0&variable=mslp&levelType=Sea%20Level&level_units=&level=Sea%20Level&timetype=4x&fileTimetype=4x&createAverage=1&year1=", 
    "&month1=1&day1=1&hr1=00%20Z&year2=",
    "&month2=12&day2=31&hr2=18%20Z&region=Custom&area_north=60&area_west=-150&area_east=-90&area_south=20&centerLat=0.0&centerLon=270.0"
]

for year in range(2003, 2024):
    id = input("Enter ID from URL (example: '2684&fileID=118848'): ")
    
    url = urlformat[0] + id + urlformat[1] + str(year) + urlformat[2] + str(year) + urlformat[3]
    
    with open('getmslp.sh', 'a') as f:
        f.write(f'curl -o mslp{year}.nc "{url}"\n')