-- 创建数据库
create database if not exists db_face_recognization;
use db_face_recognization;

----------------------------------------------------
-- 特征表
drop table if exists tb_face_feature;  -- 已废弃
create table if not exists tb_face_feature (
    person_id smallint unsigned not null auto_increment comment "person_id",
    person_name varchar(128) default null comment "person_name",
    img varchar(256) default null comment "图片路径",
    feature blob default null comment "特征值",
    create_time timestamp not null default current_timestamp comment "创建时间",
    primary key (person_id)
) engine=InnoDB auto_increment=1 default charset=utf8mb4 comment="人脸特征表";




drop table if exists tb_admin_user;
create table if not exists tb_admin_user (
    user_id int unsigned not null auto_increment comment "user_id",
    user_name varchar(64) default null comment "名字",
    pass_wd varchar(32) default null comment "密码",
    role tinyint unsigned not null default 1 comment "1 管理员，目前只有一个",
    create_time timestamp not null default current_timestamp comment "创建时间",
    primary key (user_id),
    index (user_name)
) engine=InnoDB default charset=utf8mb4 comment="管理员用户表";
insert into tb_admin_user set user_name="admin", pass_wd=md5("admin");

---------------------------------------------
-- 特征表
drop table if exists tb_multiface_feature;
create table if not exists tb_multiface_feature (
    feature_id smallint unsigned not null auto_increment comment "feature_id",
    person_id smallint unsigned not null comment "person_id",
    feature_mark varchar(128) default 'A' comment "feature_mark",
    img varchar(256) default null comment "图片路径",
    feature blob default null comment "特征值",
    create_time timestamp not null default current_timestamp comment "创建时间",
    primary key (feature_id)
) engine=InnoDB auto_increment=1 default charset=utf8mb4 comment="人脸特征表(同一个人可以有多个人脸特征)";


drop table if exists tb_person;
create table if not exists tb_person (
    person_id smallint unsigned not null auto_increment comment "person_id",
    person_name varchar(128) default null comment "person_name",
    create_time timestamp not null default current_timestamp comment "创建时间",
    primary key (person_id)
) engine=InnoDB auto_increment=1 default charset=utf8mb4 comment="人物表";


