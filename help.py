#

#download list of contig data from patric
def patric_download(alist_of_genome_ids):
  data = {}
  for genome in alist_of_genome_ids:
     print genome
     response = urllib2.urlopen('https://www.patricbrc.org/api/genome_sequence/?genome_id='+genome+'&http_accept=application/json')
     html = response.read()
     hh = json.loads(html)
     if hh:
       data[genome] = hh[0]['sequence']
     else: 
       data[genome] = 'FAIL'
  return data
