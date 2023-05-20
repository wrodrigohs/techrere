#import das bibliotecas
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
from datetime import timedelta

#remoção de warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def getNews(num_pages):
  #criação do dataframe
  df = pd.DataFrame(columns=['date' 'text', 'class'])

  #serão lidas 100 páginas
  for i in range(1, num_pages+1):
    #esse if é responsável por definir qual é a página atual
    url = 'https://exame.com/noticias-sobre/itau/' if i == 1 else f'https://exame.com/noticias-sobre/itau/{i}/'
    #essa linha verifica a resposta da requisição à página
    with requests.get(url) as page:
    #variável com o conteúdo da página
      soup = BeautifulSoup(page.text, 'lxml')

      #variável que armazena os títulos das notícias
      postings = soup.find_all('h2', 'sc-c4a67c2e-1 ftIIQp')
      print(f"\nBaixando mais {len(postings)} notícias.\n")

      #iteração sobre counter e post
      for counter, post in enumerate(postings):
          try:
              link_full = post.find('a').get('href')
              link_full = f'https://exame.com{link_full}'
                              
              url2 = link_full

              #essa linha verifica a resposta da requisição à página
              with requests.get(url2) as page2:
                soup2 = BeautifulSoup(page2.text, 'lxml')

                data = str(soup2.find('span', class_='sc-c4a67c2e-5 gpIHzq').find('p')).split('<!-- -->')[1]
                newDate = parse_pt_date(data)
                
                textContent = ""
                textPieces = soup2.find('div', class_='sc-7be271c6-1 iNMSzh').find_all('p')

                for paragraph in textPieces:
                    textContent = textContent + " " + paragraph.text
                          
                df = df.append({'date': newDate, 
                  'text':textContent}, 
                  ignore_index = True)
                if(newDate < datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')):
                  print('é menor')
                  break
              
                counter += 1
              
          except Exception as e:
              print(f'Erro: {e}')
    print(f"Notícias coletadas até agora: {len(df)}")
    print('============ + ============')
    # diff_date = datetime.datetime.now() - timedelta(days=7) 
    if(df.date[len(df) -1] < datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')):
      print('é menor')
      break

  return df

def parse_pt_date(date_string):
    '''Parses a date-time string and return datetime object
       The format is like this:
       'Seg, 21 Out 2013 22:14:36 -0200'
    '''
    FULL_MONTHS = {'janeiro': 1,  'fevereiro': 2, u'março': 3,    'abril': 4,
               'maio': 5,     'junho': 6,     'julho': 7,     'agosto': 8,
               'setembro': 9, 'outubro': 10,  'novembro': 11, 'dezembro': 12}

    date_info = date_string.lower().split()
    if ',' in date_info[0]:
        date_string = date_string.split(',')[1]
    date_info = date_string.lower().replace('de ', '').split()
    day, month_pt, year = date_info    
    month = FULL_MONTHS[month_pt]
    date_iso = '{}-{:02d}-{:02d}'.format(year, int(month), int(day))
    date_object = datetime.datetime.strptime(date_iso, '%Y-%m-%d')
    return date_object

if __name__ == '__main__':
  df = getNews(num_pages = 50)
  df = df[['date', 'class', 'text']]
  df = df.sort_values(by = 'date', ascending = False)
  df.to_csv('df.csv')