```markdown
# مستندات مجموعه داده (Data Documentation) — مسیر ج: iNaturalist

## ۱. منشأ داده و مجوز استفاده
داده‌های این پروژه از مجموعه داده باز **iNaturalist Open Data** (در دسترس از طریق AWS Registry of Open Data و API رسمی iNaturalist v1) استخراج شده‌اند. مشاهدات انتخابی دارای درجه کیفی `research` (تأییدشده توسط جامعه علمی و حداقل دو متخصص/کاربر مستقل) هستند.

## ۲. ساختار فایل‌ها و جداول
پس از اجرای اسکریپت `src/data_fetcher.py`، فایل‌های زیر تولید می‌شوند:

1. `data/species_toxicity_metadata.csv`: جدول مرجع ۴۰ گونه قارچ شامل نام علمی (`scientific_name`)، نام عمومی (`common_name`)، خانواده زیستی (`family`)، کلاس سمیت (`toxicity_class`: شامل `Deadly_Toxic`، `Inedible_MildToxic`، `Edible`)، سطح ریسک عددی (`risk_level`) و گروه شباهت ظاهری (`lookalike_group`).
2. `data/raw/fungi_observations_raw.csv`: جدول مشاهدات خام شامل شناسه مشاهده (`observation_uuid`)، شناسه گونه (`taxon_id`)، شناسه مشاهده‌گر (`observer_id`)، تاریخ و زمان مشاهده (`observed_on`)، مختصات جغرافیایی (`latitude`, `longitude`)، دقت مکانی (`positional_accuracy`)، قاره (`region`) و تعداد تصاویر ثبت‌شده برای آن مشاهده (`num_photos`).
3. `data/raw/fungi_photos_raw.csv`: جدول تصاویر مرتبط با مشاهدات (رابطه یک‌به‌چند از `observation_uuid` به `photo_id`) شامل آدرس تصویر، ابعاد و مجوز تصویر.
4. `data/processed/phase1_audited_observations.parquet`: داده خروجی فاز اول پس از ممیزی کیفیت، برچسب‌گذاری رکوردهای معیوب/پرت و استخراج متغیرهای پایه زمانی و مکانی.

## ۳. بازتولید داده
برای دانلود یا بازسازی دقیق داده‌ها از ریشه مخزن دستور زیر را اجرا کنید:
```bash
python src/data_fetcher.py --config configs/phase1_config.yaml --mode auto