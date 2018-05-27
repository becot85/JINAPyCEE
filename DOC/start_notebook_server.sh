#set the path to your NUPYCEE dir
cd ../
export SYGMADIR=`pwd`
cd -
echo 'set SYGMADIR to '$SYGMADIR

#export path
export PYTHONPATH=$PYTHONPATH:$SYGMADIR

#starts notebook server
jupyter notebook

