cd backup

ls_date=`date +%Y%m%d`
rm -rf ${ls_date}
mkdir ${ls_date}
cd ${ls_date}
cp ../../*.py ./
cp ../../*.sh ./

cd ..
rm -f ${ls_date}.tar
tar -cf ${ls_date}.tar ${ls_date}
rm -rf ${ls_date}

